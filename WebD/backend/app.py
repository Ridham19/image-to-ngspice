"""
WebSpice Studio — FastAPI Backend
Endpoints:
  POST /api/detect     — Upload image, run YOLO + OCR, return components
  POST /api/simulate   — Receive canvas state, generate SPICE netlist, run ngspice
  GET  /api/plots/{fn} — Serve generated plot images
"""
import os
import sys
import shutil
import json
import subprocess
import re
import glob
from collections import defaultdict
from typing import List, Dict, Any, Optional


# ═══════════════════════════════════════════
# FILENAME SANITIZATION HELPER
# ═══════════════════════════════════════════
def _safe_filename(name: str) -> str:
    """
    Sanitize an uploaded filename to be safe for local filesystem storage.

    Replaces whitespace and any character outside [A-Za-z0-9._-] with
    underscores, collapses consecutive underscores, and strips leading
    and trailing dots / underscores to prevent path-traversal vectors.

    Parameters
    ----------
    name : str
        Original filename as received from the multipart upload.

    Returns
    -------
    str -- Sanitized filename (basename only, no directory component).
    """
    # Take only the basename to strip any path-traversal prefix
    name = os.path.basename(name)
    # Replace whitespace and forbidden characters with underscores
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    # Collapse consecutive underscores
    name = re.sub(r"_+", "_", name)
    # Strip leading/trailing dots and underscores
    name = name.strip("._")
    # Fallback for edge-case empty result
    return name or "upload"

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

# Import ML Logic
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.model import ComponentDetector
from core.node_solver import solve_canvas, NON_DEVICE_TYPES

# ═══════════════════════════════════════════
# APP INIT
# ═══════════════════════════════════════════
app = FastAPI(title="Image-to-SPICE API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════
# WORKSPACE DIRECTORIES
# ═══════════════════════════════════════════
WORKSPACE_DIR = os.path.join(os.path.dirname(__file__), "workspace")
UPLOAD_DIR = os.path.join(WORKSPACE_DIR, "uploads")
SIM_DIR = os.path.join(WORKSPACE_DIR, "simulations")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SIM_DIR, exist_ok=True)

# ═══════════════════════════════════════════
# CONFIGURATION LOAD (NGSPICE & MODEL PATHS)
# ═══════════════════════════════════════════
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")
if not os.path.exists(CONFIG_FILE):
    CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.json")

NGSPICE_PATH = "ngspice"  # fallback: expect on system PATH

if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, "r") as f:
            cfg = json.load(f)
            NGSPICE_PATH = cfg.get("ngspice_path", NGSPICE_PATH)
        print(f"📋 Loaded config from {CONFIG_FILE}: ngspice_path='{NGSPICE_PATH}'")
    except Exception as e:
        print(f"⚠️ Could not read config.json: {e}")

# ═══════════════════════════════════════════
# COMPONENT DATABASE (Server-side mirror)
# Matches frontend COMPONENT_DB & PySpice_studio/library.py
# ═══════════════════════════════════════════
GRID_SIZE = 20

SPICE_TEMPLATES = {
    'resistor':       '{name} {n1} {n2} {value}',
    'resistor_photo': '{name} {n1} {n2} {value}',
    'capacitor':      '{name} {n1} {n2} {value} ic={ic}',
    'capacitor_polarized': '{name} {n1} {n2} {value} ic={ic}',
    'inductor':       '{name} {n1} {n2} {value} ic={ic}',
    'diode':          '{name} {n1} {n2} {model}',
    'diode_led':      '{name} {n1} {n2} {model}',
    'diode_zener':    '{name} {n1} {n2} {model}',
    'source':         '{name} {n1} {n2} DC {dc}',
    'voltage_source': '{name} {n1} {n2} DC {dc}',
    'current_source': '{name} {n1} {n2} DC {dc}',
    'vss':            '{name} {n1} {n2} DC {dc}',
    'ac_source':      '{name} {n1} {n2} AC {mag} {phase}',
    'pulse_source':   '{name} {n1} {n2} PULSE({v1} {v2} {td} {tr} {tf} {pw} {per})',
    'sine_source':    '{name} {n1} {n2} SINE({vo} {va} {freq} {td} {theta} {phase})',
    'exp_source':     '{name} {n1} {n2} EXP({v1} {v2} {td1} {tau1} {td2} {tau2})',
    'pwl_source':     '{name} {n1} {n2} PWL({pwl_data})',
    'sffm_source':    '{name} {n1} {n2} SFFM({vo} {va} {fc} {mdi} {fs})',
    'am_source':      '{name} {n1} {n2} AM({va} {fc} {mf} {ph})',
    'bjt_npn':        '{name} {n2} {n1} {n3} {model}',
    'bjt_pnp':        '{name} {n2} {n1} {n3} {model}',
    'bjt':            '{name} {n2} {n1} {n3} {model}',
    'mosfet':         '{name} {n2} {n1} {n3} {n3} {model} w={w} l={l}',
    'nmos':           '{name} {n2} {n1} {n3} {n3} {model} w={w} l={l}',
    'pmos':           '{name} {n2} {n1} {n3} {n3} {model} w={w} l={l}',
    'phototransistor': '{name} {n2} {n1} {n3} {model}',
    # ── Subcircuit / IC types — handled specially in simulate endpoint ──
    # 'opamp'       → emits: X{name} {n_vplus} {n_vminus} {n_out} {vs_pos_node} {vs_neg_node} {model}
    # 'ic'          → emits: X{name} <node list from num_pins> {subckt_name}
    # 'transformer' → emits: L{name}a {n1} {n2} {value}  +  L{name}b {n3} {n4} {value}  +  K{name} ...
}

# ═══════════════════════════════════════════
# SUBCIRCUIT DEFINITIONS
# Built-in .subckt blocks prepended to the netlist when the matching
# component type appears on the canvas.  Keys match component types.
# ═══════════════════════════════════════════
SUBCKT_DEFINITIONS: Dict[str, str] = {
    # Ideal op-amp model (5 external pins: V+, V−, OUT, VS+, VS−)
    # Based on a simple voltage-controlled voltage source (VCVS) approximation.
    'opamp': """\
* --- Ideal Op-Amp Subcircuit: LM741 ---
.subckt LM741 vplus vminus out vspos vsneg
* Input resistance between inputs
Rin vplus vminus 2MEG

* Controlled voltage source referenced to ground at an internal node
Eout int 0 vplus vminus 200000

* Output resistance connected between the internal node and the output pin
Rout_int int out 75
.ends LM741""",

    # Internally no fixed definition — user supplies .subckt_name param
    # and the definition is expected to be provided externally or via the
    # custom_subckt param field.  We emit a placeholder comment.
    'ic': None,

    # Transformer: no .subckt needed — uses coupled inductors directly
    'transformer': None,
}

# SPICE prefix order for sorted netlist output
# X = subcircuit instances (op-amps, ICs, etc.); K = mutual inductance (transformers)
SPICE_PREFIX_ORDER = {'C': 0, 'D': 1, 'I': 2, 'K': 3, 'L': 4, 'Q': 5, 'R': 6, 'V': 7, 'X': 8}

# ═══════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════
class WirePoint(BaseModel):
    x: float
    y: float

class ComponentPayload(BaseModel):
    type: str
    name: str
    x: float
    y: float
    params: Dict[str, str] = {}
    rotation: int = 0

class SimConfig(BaseModel):
    """
    Simulation configuration matching PySpice Studio's SimulationDialog.

    Fields
    ------
    mode   : "op" | "tran" | "dc" | "ac"
    params : mode-specific parameters (step, stop, start, source, etc.)
    plots  : dict mapping window_id → list of signal expressions
             e.g. {"1": ["v(1)", "v(2)"], "2": ["i(V1)"]}
    colors : dict mapping color index → color name
             e.g. {"0": "white", "1": "black", "2": "red"}
    """
    mode: str = "op"
    params: Dict[str, str] = {}
    plots: Dict[str, List[str]] = {}
    colors: Dict[str, str] = {}

class SimulateRequest(BaseModel):
    components: List[ComponentPayload]
    wires: List[List[WirePoint]]
    simConfig: SimConfig = SimConfig()
    custom_netlist: Optional[str] = None

# ═══════════════════════════════════════════
# LOAD ML MODEL
# ═══════════════════════════════════════════
print("🚀 Booting up API Server...")
detector = ComponentDetector(model_name="../../epoch_40.pt")

# ═══════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════

@app.get("/")
def read_root():
    return {"status": "Online", "message": "Welcome to the Image-to-SPICE Engine"}


@app.post("/api/detect")
async def detect_circuit(file: UploadFile = File(...)):
    """
    Upload a circuit image, run YOLO + OCR component detection, then execute
    the full wire-tracing vision pipeline to yield a connectivity matrix.

    Pipeline
    --------
    1. Sanitize filename and save upload to workspace/uploads/
    2. Confirm OpenCV can read the saved file
    3. Run YOLO + EasyOCR detection on the preloaded image matrix
    4. Preprocess image with dual-path binarization (adaptive + Canny)
    5. Separate wire layer by BBOX collision masking
    6. DFS pixel-graph traversal -> node connectivity list
    7. Return JSON with components + connections (null on tracing failure)
    """
    try:
        # Step 1 -- Sanitize filename and resolve cross-platform save path
        safe_name = _safe_filename(file.filename or "upload.png")
        file_location = os.path.join(UPLOAD_DIR, safe_name)

        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        saved_bytes = os.path.getsize(file_location)
        print(f"📁 Saved upload: {file_location!r} ({saved_bytes} bytes)")

        # Step 2 -- Verify OpenCV can decode the saved file before any ML work
        import cv2 as _cv2
        bgr_image = _cv2.imread(file_location)
        if bgr_image is None:
            return JSONResponse(
                status_code=422,
                content={
                    "status": "error",
                    "message": (
                        f"The uploaded file '{safe_name}' could not be decoded as an image. "
                        "Please upload a valid PNG, JPG, or BMP file."
                    ),
                },
            )

        print(f"🖼️  Image decoded: {bgr_image.shape[1]}x{bgr_image.shape[0]} px")

        # Step 3 -- YOLO + OCR detection
        # Pass the preloaded BGR matrix to avoid a second disk read inside detect()
        json_output_path = os.path.join(WORKSPACE_DIR, "latest_detection.json")
        detected_comps = detector.detect(bgr_image, output_file=json_output_path)
        print(f"🔍 Detected {len(detected_comps)} component(s)")

        # Steps 4-6 -- Wire Tracing Pipeline
        connections = []
        debug_image_b64 = ""
        pin_anchors = []  # Will be populated by wire tracing; sent to frontend for interactive editing
        try:
            from core.processing import compute_pin_anchors, TRANSPARENT_TYPES
            from wire_tracer.tracer import trace_wires
            import cv2, base64

            # Step 4a: Compute pin anchors using the core dual-path preprocessor
            # for high-quality rotation calibration (this preprocessor is
            # validated on hand-drawn circuit images).
            from core.processing import preprocess_image as core_preprocess
            _, _, core_binary = core_preprocess(bgr_image)
            pin_anchors = compute_pin_anchors(detected_comps, wire_mask=core_binary)
            
            # Step 4b: Transform detected_comps to wire_tracer format
            wt_components = []
            for idx, comp in enumerate(detected_comps):
                ctype = comp.get("type", "")
                if ctype in TRANSPARENT_TYPES:
                    continue
                
                wt_comp = {
                    "id": f"comp_{idx}",
                    "label": ctype,
                    "bbox": comp.get("box", [0,0,0,0]),
                    "pins": []
                }
                
                for anchor in pin_anchors:
                    if anchor["comp_idx"] == idx:
                        wt_comp["pins"].append({
                            "id": f"{idx}_{anchor['pin_id']}",
                            "loc": [anchor["x"], anchor["y"]]
                        })
                
                wt_components.append(wt_comp)
                
            # Step 5: Run the wire_tracer pipeline (has its own preprocessing)
            netlist, debug_info = trace_wires(
                bgr_image, 
                wt_components, 
                method="connected_components", 
                debug=True
            )
            
            # Assign nodes to detected_comps for frontend state
            for comp_idx, det in enumerate(detected_comps):
                ctype = det.get("type", "")
                if ctype in TRANSPARENT_TYPES:
                    continue
                pin_list = [p for p in pin_anchors if p["comp_idx"] == comp_idx]
                nodes_list = []
                for p in sorted(pin_list, key=lambda k: k["pin_id"]):
                    pin_id_str = f"{comp_idx}_{p['pin_id']}"
                    net_label = debug_info["pin_net_map"].get(pin_id_str)
                    nodes_list.append(str(net_label) if net_label is not None else "NC")
                det["nodes"] = nodes_list

            # Build sequential connections for the frontend A* router
            for net_pins in netlist:
                for i in range(len(net_pins) - 1):
                    p1_idx, p1_pid = map(int, net_pins[i].split("_"))
                    p2_idx, p2_pid = map(int, net_pins[i+1].split("_"))
                    connections.append({
                        "pin1": {"comp_idx": p1_idx, "pin_id": p1_pid},
                        "pin2": {"comp_idx": p2_idx, "pin_id": p2_pid}
                    })
            
            # Generate Debug Image
            from wire_tracer.utils import draw_debug_overlay
            debug_overlay_bgr = draw_debug_overlay(
                bgr_image, 
                wt_components, 
                debug_info["label_map"], 
                debug_info["pin_net_map"]
            )
            success, encoded_img = cv2.imencode('.jpg', debug_overlay_bgr)
            if success:
                debug_image_b64 = "data:image/jpeg;base64," + base64.b64encode(encoded_img).decode('utf-8')

            print(f"✅ Wire tracing complete: {len(connections)} logical wire segment(s) found across {debug_info.get('num_nets', 0)} nets")

        except Exception as wire_err:
            import traceback
            traceback.print_exc()
            print(
                f"⚠️  Wire tracing failed for '{safe_name}' "
                f"({type(wire_err).__name__}: {wire_err}) — returning connections: null"
            )
            # connections stays None; the API returns a partial success

        # Step 7 -- Serialize and return
        safe_components = jsonable_encoder(detected_comps)
        safe_connections = jsonable_encoder(connections)
        safe_pin_anchors = jsonable_encoder(pin_anchors)

        return JSONResponse(content={
            "status": "success",
            "components": safe_components,
            "connections": safe_connections,
            "pin_anchors": safe_pin_anchors,
            "debug_image": debug_image_b64,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )


@app.post("/api/simulate")
async def simulate_circuit(request: SimulateRequest):
    """
    Receive canvas state, generate SPICE netlist via DFS graph traversal,
    write to file, run ngspice, return results + plot images.
    """
    try:
        # Define simulation config defaults to prevent UnboundLocalError in step 8
        sim_mode = request.simConfig.mode.lower() if request.simConfig.mode else "op"
        plots = request.simConfig.plots or {}
        colors = request.simConfig.colors or {}

        # ─── STEP 1: Solve node topology via DFS ───
        node_map = {}
        comp_pins = []
        try:
            node_map, comp_pins = solve_canvas(
                request.components, request.wires, GRID_SIZE
            )
        except Exception as e:
            if not request.custom_netlist:
                raise e

        if request.custom_netlist:
            netlist_text = request.custom_netlist
        else:
            lines = ["* WebSpice Studio — Generated Netlist", ""]
    
            # Collect required .model statements and .subckt definitions
            models_needed = set()
            subckts_needed: Dict[str, str] = {}  # type_key -> subckt text block
    
            for comp, _ in comp_pins:
                if comp.type == 'diode':
                    models_needed.add(".Model Dx diode (Is=14n Rs=0 N=1)")
                elif comp.type == 'diode_led':
                    models_needed.add(".Model Dx diode (Is=14n Rs=0 N=1)")
                elif comp.type == 'diode_zener':
                    models_needed.add(".Model Dx diode (Is=14n Rs=0 N=1 BV=5.1)")
                elif comp.type in ('bjt_npn', 'bjt', 'phototransistor'):
                    models_needed.add(".Model Tx NPN (BF=300)")
                elif comp.type == 'bjt_pnp':
                    models_needed.add(".Model Tx_pnp PNP (BF=300)")
                elif comp.type in ('mosfet', 'nmos'):
                    model_param = comp.params.get('model', 'Mx' if comp.type == 'mosfet' else 'nmos')
                    models_needed.add(f".model {model_param} nmos level=54")
                elif comp.type == 'pmos':
                    model_param = comp.params.get('model', 'pmos')
                    models_needed.add(f".model {model_param} pmos level=54")
                elif comp.type == 'opamp':
                    # Only embed the built-in LM741 definition if no custom subckt provided
                    model_name = comp.params.get('model', 'LM741')
                    if model_name == 'LM741' and 'opamp' not in subckts_needed:
                        defn = SUBCKT_DEFINITIONS.get('opamp')
                        if defn:
                            subckts_needed['opamp'] = defn
                elif comp.type == 'ic':
                    # User-defined subcircuit: embed custom_subckt body if supplied
                    custom_body = comp.params.get('custom_subckt', '').strip()
                    subckt_name = comp.params.get('subckt_name', 'MyIC')
                    key = f'ic_{subckt_name}'
                    if custom_body and key not in subckts_needed:
                        subckts_needed[key] = f"\n* --- User-defined subcircuit: {subckt_name} ---\n{custom_body}"
    
            # Prepend .subckt definitions (before .model statements)
            for defn_text in subckts_needed.values():
                lines.append(defn_text)
            if subckts_needed:
                lines.append("")
    
            for m in sorted(models_needed):
                lines.append(m)
            if models_needed:
                lines.append("")
    
            # Generate device lines — sorted by SPICE prefix character
            device_lines = []
            for comp, pins in comp_pins:
                if comp.type in NON_DEVICE_TYPES:
                    continue
    
                # Look up node IDs for this component's pins
                nodes = [node_map.get(pin, "NC") for pin in pins]
    
                # ── Op-Amp: X-prefix subcircuit instantiation ──────────────────
                if comp.type == 'opamp':
                    n_vplus  = nodes[0] if len(nodes) > 0 else 'NC'
                    n_vminus = nodes[1] if len(nodes) > 1 else 'NC'
                    n_out    = nodes[2] if len(nodes) > 2 else 'NC'
                    vs_pos   = comp.params.get('vs_pos', '15')
                    vs_neg   = comp.params.get('vs_neg', '-15')
                    model    = comp.params.get('model', 'LM741')
                    # Supply rails: create implicit DC supply nodes named after instance
                    vs_pos_node = f"vsp_{comp.name}"
                    vs_neg_node = f"vsn_{comp.name}"
                    # Supply voltage sources for the op-amp rails
                    device_lines.append((SPICE_PREFIX_ORDER.get('V', 7), f"VsPos_{comp.name}",
                        f"VsPos_{comp.name} {vs_pos_node} 0 DC {vs_pos}"))
                    device_lines.append((SPICE_PREFIX_ORDER.get('V', 7), f"VsNeg_{comp.name}",
                        f"VsNeg_{comp.name} {vs_neg_node} 0 DC {vs_neg}"))
                    # Subcircuit instantiation
                    line = f"X{comp.name} {n_vplus} {n_vminus} {n_out} {vs_pos_node} {vs_neg_node} {model}"
                    device_lines.append((SPICE_PREFIX_ORDER.get('X', 8), f"X{comp.name}", line))
                    continue
    
                # ── Generic IC: X-prefix subcircuit with dynamic node list ─────
                if comp.type == 'ic':
                    subckt_name = comp.params.get('subckt_name', 'MyIC')
                    node_list = " ".join(nodes) if nodes else "NC"
                    line = f"X{comp.name} {node_list} {subckt_name}"
                    device_lines.append((SPICE_PREFIX_ORDER.get('X', 8), f"X{comp.name}", line))
                    continue
    
                # ── Transformer: coupled inductor pair + K statement ───────────
                if comp.type == 'transformer':
                    n1 = nodes[0] if len(nodes) > 0 else 'NC'
                    n2 = nodes[1] if len(nodes) > 1 else 'NC'
                    n3 = nodes[2] if len(nodes) > 2 else 'NC'
                    n4 = nodes[3] if len(nodes) > 3 else 'NC'
                    inductance = comp.params.get('value', '1m')
                    coupling   = comp.params.get('coupling', '0.99')
                    la_name = f"L{comp.name}a"
                    lb_name = f"L{comp.name}b"
                    k_name  = f"K{comp.name}"
                    device_lines.append((SPICE_PREFIX_ORDER.get('L', 4), la_name,
                        f"{la_name} {n1} {n2} {inductance}"))
                    device_lines.append((SPICE_PREFIX_ORDER.get('L', 4), lb_name,
                        f"{lb_name} {n3} {n4} {inductance}"))
                    device_lines.append((SPICE_PREFIX_ORDER.get('K', 3), k_name,
                        f"{k_name} {la_name} {lb_name} {coupling}"))
                    continue
    
                # ── Standard template-based device ────────────────────────────
                template = SPICE_TEMPLATES.get(comp.type)
                if not template:
                    continue
    
                # Build context dict for template formatting
                ctx = {
                    'name': comp.name,
                    'n1': nodes[0] if len(nodes) > 0 else '0',
                    'n2': nodes[1] if len(nodes) > 1 else '0',
                    'n3': nodes[2] if len(nodes) > 2 else '0',
                    'n4': nodes[3] if len(nodes) > 3 else '0',
                    'w': '10u',
                    'l': '0.18u',
                }
                # Merge component params
                for k, v in comp.params.items():
                    ctx[k] = v
    
                try:
                    line = template.format(**ctx)
                    # Sort key: first char of the component name (SPICE prefix)
                    prefix = comp.name[0].upper() if comp.name else 'Z'
                    sort_key = SPICE_PREFIX_ORDER.get(prefix, 99)
                    device_lines.append((sort_key, comp.name, line))
                except KeyError as e:
                    device_lines.append((99, comp.name, f"* ERROR: Missing param {e} for {comp.name}"))
    
            # Sort by SPICE prefix order, then by name
            device_lines.sort(key=lambda t: (t[0], t[1]))
            for _, _, line in device_lines:
                lines.append(line)
    
    
            # ─── STEP 3: Append simulation command ───
            lines.append("")
            sim_mode = request.simConfig.mode.lower()
            sim_params = request.simConfig.params
    
            if sim_mode == "tran":
                step = sim_params.get("step", "0.1m")
                stop = sim_params.get("stop", "80m")
                start = sim_params.get("start", "0")
                lines.append(f".tran {step} {stop} {start}")
            elif sim_mode == "dc":
                src1 = sim_params.get("source1", sim_params.get("source", "V1"))
                start1 = sim_params.get("start", "0")
                stop1 = sim_params.get("stop", "5")
                incr1 = sim_params.get("incr", "0.1")
                dc_cmd = f".dc {src1} {start1} {stop1} {incr1}"
                # Secondary sweep (optional, matches Python app)
                src2 = sim_params.get("source2", "")
                if src2 and src2.lower() != "none":
                    start2 = sim_params.get("start2", "0")
                    stop2 = sim_params.get("stop2", "5")
                    incr2 = sim_params.get("incr2", "1")
                    dc_cmd += f" {src2} {start2} {stop2} {incr2}"
                lines.append(dc_cmd)
            elif sim_mode == "ac":
                atype = sim_params.get("type", "DEC")
                pts = sim_params.get("points", "10")
                fstart = sim_params.get("fstart", "1")
                fstop = sim_params.get("fstop", "10meg")
                lines.append(f".ac {atype} {pts} {fstart} {fstop}")
            else:
                lines.append(".op")
    
            # ─── STEP 4: Control block (matches PySpice_studio/netlist.py) ───
            lines.append(".control")
            lines.append("run")
    
            # Color settings
            colors = request.simConfig.colors
            if not colors:
                lines.append("set color0 = white")
                lines.append("set color1 = black")
            else:
                for idx in sorted(colors.keys(), key=lambda k: int(k) if k.isdigit() else 999):
                    lines.append(f"set color{idx} = {colors[idx]}")
    
            lines.append("set xbrushwidth = 2")
    
            # Print all for data extraction
            lines.append("print all > sim_out.txt")
    
            # Plot commands — generate hardcopy SVG images for each plot window
            # Clean up old plot files first
            for old_file in glob.glob(os.path.join(SIM_DIR, "plot_win*.svg")):
                try:
                    os.remove(old_file)
                except OSError:
                    pass
            
            sim_out_path = os.path.join(SIM_DIR, "sim_out.txt")
            if os.path.exists(sim_out_path):
                try:
                    os.remove(sim_out_path)
                except OSError:
                    pass
    
            plots = request.simConfig.plots
            if plots and sim_mode != "op":
                lines.append("set hcopydevtype = svg")
                sorted_wins = sorted(plots.keys())
                for win_id in sorted_wins:
                    sigs = " ".join(plots[win_id])
                    svg_file = f"plot_win{win_id}.svg"
                    lines.append(f"hardcopy {svg_file} {sigs} title 'Graph Window {win_id}'")
    
            lines.append(".endc")
            lines.append(".end")
    
            netlist_text = "\n".join(lines)

        # ─── STEP 5: Write netlist to file ───
        cir_path = os.path.join(SIM_DIR, "circuit.cir")
        with open(cir_path, "w", encoding="utf-8") as f:
            f.write(netlist_text)
        print(f"📝 Netlist written to {cir_path}")

        # ─── STEP 6: Run ngspice subprocess ───
        raw_output = ""
        sim_data = {}

        # ─── STEP 6: Run ngspice subprocess ───
        raw_output = ""
        sim_data = {}
        node_map_json = {f"{pt[0]},{pt[1]}": node for pt, node in node_map.items()}

        try:
            if not os.path.exists(NGSPICE_PATH) and NGSPICE_PATH != "ngspice":
                logs = diagnose_simulation_output("", netlist_text, request.components)
                logs.append({
                    "type": "error",
                    "message": f"ngspice binary not found at: {NGSPICE_PATH}",
                    "source": "backend",
                    "component": None,
                    "node": None,
                    "line_number": None,
                    "line_text": None
                })
                return JSONResponse(content={
                    "status": "error",
                    "message": f"ngspice binary not found at: {NGSPICE_PATH}",
                    "netlist": netlist_text,
                    "raw_output": "",
                    "plot_images": [],
                    "node_map": node_map_json,
                    "logs": logs
                })

            log_path = os.path.join(SIM_DIR, "simulation.log")
            command = [NGSPICE_PATH, "-b", "-o", log_path, cir_path]

            result = subprocess.run(
                command, capture_output=True, text=True, check=False, shell=True, cwd=SIM_DIR
            )

            # Read simulation.log redirect output
            log_content = ""
            if os.path.exists(log_path):
                try:
                    with open(log_path, "r", encoding="utf-8", errors="replace") as f_log:
                        log_content = f_log.read()
                except Exception as log_err:
                    log_content = f"Error reading simulation.log: {log_err}"

            # Combine stdout, stderr, and redirected log content for diagnosis and display
            raw_output = (result.stdout or "") + "\n" + (result.stderr or "") + "\n" + log_content

            # 1. Write errors to errors.txt
            error_file = os.path.join(SIM_DIR, "errors.txt")
            if result.stderr:
                with open(error_file, "w", encoding="utf-8") as f_err:
                    f_err.write(result.stderr)
            else:
                with open(error_file, "w", encoding="utf-8") as f_err:
                    f_err.write("No errors reported by Ngspice.\n")

            # 2. Write normal output to output.log
            output_log_file = os.path.join(SIM_DIR, "output.log")
            if result.stdout:
                with open(output_log_file, "w", encoding="utf-8") as f_out:
                    f_out.write(result.stdout)

            # ─── STEP 7: Parse ngspice output ───
            sim_out_path = os.path.join(SIM_DIR, "sim_out.txt")
            if os.path.exists(sim_out_path):
                with open(sim_out_path, "r", encoding="utf-8") as f:
                    file_output = f.read()
                sim_data = parse_ngspice_output(file_output)
            else:
                sim_data = parse_ngspice_output(raw_output)

            if result.returncode != 0 and not sim_data:
                logs = diagnose_simulation_output(raw_output, netlist_text, request.components)
                logs.append({
                    "type": "error",
                    "message": f"ngspice exited with code {result.returncode}",
                    "source": "backend",
                    "component": None,
                    "node": None,
                    "line_number": None,
                    "line_text": None
                })
                return JSONResponse(content={
                    "status": "error",
                    "message": f"ngspice exited with code {result.returncode}",
                    "netlist": netlist_text,
                    "raw_output": "",
                    "plot_images": [],
                    "node_map": node_map_json,
                    "logs": logs
                })

        except FileNotFoundError:
            logs = diagnose_simulation_output("", netlist_text, request.components)
            logs.append({
                "type": "error",
                "message": f"ngspice executable not found. Tried: {NGSPICE_PATH}.",
                "source": "backend",
                "component": None,
                "node": None,
                "line_number": None,
                "line_text": None
            })
            return JSONResponse(content={
                "status": "error",
                "message": f"ngspice executable not found. Tried: {NGSPICE_PATH}. Install ngspice or update config.json.",
                "netlist": netlist_text,
                "raw_output": "",
                "plot_images": [],
                "node_map": node_map_json,
                "logs": logs
            })

        # ─── STEP 8: Collect generated plot images ───
        plot_images = []

        # Check for SVG files generated by ngspice hardcopy
        for svg_path in sorted(glob.glob(os.path.join(SIM_DIR, "plot_win*.svg"))):
            filename = os.path.basename(svg_path)
            plot_images.append(f"/api/plots/{filename}")

        # If ngspice didn't generate SVGs (older version / no hardcopy support),
        # fall back to matplotlib for server-side plot generation
        if not plot_images and plots and sim_mode != "op":
            fallback_images = generate_plot_images_matplotlib(
                sim_data, plots, colors, SIM_DIR, sim_mode
            )
            for fname in fallback_images:
                plot_images.append(f"/api/plots/{fname}")

        # Build the unique node list for the frontend
        unique_nodes = sorted(set(node_map.values()))

        # Build the source list for the frontend (sources available for sweep)
        source_names = []
        sweepable_names = []
        source_types = {'source', 'voltage_source', 'current_source', 'vss', 'ac_source', 'pulse_source', 'sine_source', 'exp_source', 'pwl_source', 'sffm_source', 'am_source'}
        sweepable_types = source_types | {'resistor', 'resistor_photo'}
        for comp, _ in comp_pins:
            if comp.type in source_types:
                source_names.append(comp.name)
            if comp.type in sweepable_types:
                sweepable_names.append(comp.name)

        logs = diagnose_simulation_output(raw_output, netlist_text, request.components)

        return JSONResponse(content={
            "status": "success",
            "netlist": netlist_text,
            "raw_output": "",
            "data": sim_data,
            "plot_images": plot_images,
            "nodes": unique_nodes,
            "sources": source_names,
            "sweepables": sweepable_names,
            "node_map": node_map_json,
            "logs": logs
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        logs = []
        if 'raw_output' in locals():
            logs = diagnose_simulation_output(raw_output, netlist_text if 'netlist_text' in locals() else "", request.components)
        logs.append({
            "type": "error",
            "message": f"Topology / Netlist Generation Error: {str(e)}",
            "source": "backend",
            "component": None,
            "node": None,
            "line_number": None,
            "line_text": None
        })
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": str(e),
            "netlist": netlist_text if 'netlist_text' in locals() else "",
            "raw_output": "",
            "plot_images": [],
            "node_map": node_map_json if 'node_map_json' in locals() else {},
            "logs": logs
        })


@app.get("/api/simulation_log")
async def get_simulation_log():
    """Fetch the contents of the simulation log file."""
    log_path = os.path.join(SIM_DIR, "simulation.log")
    if not os.path.exists(log_path):
        return JSONResponse(status_code=404, content={"error": "Simulation log not found"})
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return {"content": content}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/plots/{filename}")
async def get_plot_image(filename: str):
    """Serve a generated plot image (SVG or PNG) from the simulations directory."""
    filepath = os.path.join(SIM_DIR, filename)
    if not os.path.exists(filepath):
        return JSONResponse(status_code=404, content={"error": "Plot not found"})

    # Determine media type
    if filename.endswith(".svg"):
        media_type = "image/svg+xml"
    elif filename.endswith(".png"):
        media_type = "image/png"
    else:
        media_type = "application/octet-stream"

    return FileResponse(filepath, media_type=media_type)


@app.post("/api/solve_nodes")
async def solve_nodes_endpoint(request: SimulateRequest):
    """
    Pre-simulation helper: solve the node topology and return available
    nodes, sources, and sweepable components for the simulation dialog.
    """
    try:
        node_map, comp_pins = solve_canvas(
            request.components, request.wires, GRID_SIZE
        )

        unique_nodes = sorted(set(node_map.values()))
        node_map_json = {f"{pt[0]},{pt[1]}": node for pt, node in node_map.items()}

        source_names = []
        sweepable_names = []
        source_types = {'source', 'voltage_source', 'current_source', 'vss', 'ac_source', 'pulse_source', 'sine_source', 'exp_source', 'pwl_source', 'sffm_source', 'am_source'}
        sweepable_types = source_types | {'resistor', 'resistor_photo'}
        for comp, _ in comp_pins:
            if comp.type in source_types:
                source_names.append(comp.name)
            if comp.type in sweepable_types:
                sweepable_names.append(comp.name)

        return JSONResponse(content={
            "status": "success",
            "nodes": unique_nodes,
            "sources": source_names,
            "sweepables": sweepable_names,
            "node_map": node_map_json,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": str(e),
            "logs": [{
                "type": "error",
                "message": f"Topology / Node Solver Error: {str(e)}",
                "source": "backend",
                "component": None,
                "node": None,
                "line_number": None,
                "line_text": None
            }]
        })


# ═══════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════

def parse_ngspice_output(raw: str) -> dict:
    """
    Parse ngspice stdout text for data vectors.
    Handles both 'print all' tabular output and basic operating point output.
    Returns dict with 'vectors' and 'values' keys.
    """
    result = {"vectors": [], "values": []}

    # Try to parse 'print all' output lines
    # ngspice prints lines like: v(1) = 5.000000e+00
    value_pattern = re.compile(r'^\s*([\w()]+)\s*=\s*([+-]?[\d.eE+\-]+)', re.MULTILINE)
    matches = value_pattern.findall(raw)

    if matches:
        for name, val in matches:
            try:
                result["vectors"].append(name)
                result["values"].append(float(val))
            except ValueError:
                pass

    # Also try to capture tabular data (from .tran, .dc, .ac)
    # ngspice tabular format: Index   time   v(1)   v(2)  ...
    table_header = re.search(r'^Index\s+(.+)$', raw, re.MULTILINE)
    if table_header:
        cols = table_header.group(1).split()
        result["vectors"] = cols
        result["values"] = []

        # Grab data lines following the header (with dashes separator)
        lines = raw.split('\n')
        data_started = False
        for line in lines:
            if re.match(r'^-+$', line.strip()):
                data_started = True
                continue
            if data_started:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        row = [float(p) for p in parts[1:]]  # skip index column
                        result["values"].append(row)
                    except ValueError:
                        data_started = False

    return result


def diagnose_simulation_output(raw_output: str, netlist: str, components: list) -> list:
    """
    Parse ngspice stdout/stderr text and generated netlist for warnings/errors.
    Maps them to specific components, nodes, and netlist line numbers.
    """
    logs = []
    lines = raw_output.split('\n') if raw_output else []
    netlist_lines = netlist.split('\n') if netlist else []

    def find_netlist_line(pattern: str, check_contains=True):
        for idx, line in enumerate(netlist_lines):
            if check_contains:
                if pattern.lower() in line.lower():
                    return idx + 1, line
            else:
                words = re.findall(r'\b\w+\b', line)
                if any(w.lower() == pattern.lower() for w in words):
                    return idx + 1, line
        return None, None

    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            continue

        log_type = None
        message = ""
        source = "ngspice"

        line_lower = line_strip.lower()
        if "fatal error" in line_lower:
            log_type = "error"
            message = line_strip
        elif "error" in line_lower or "singular matrix" in line_lower or "check node" in line_lower or "fail" in line_lower or "missing" in line_lower or "unknown device" in line_lower:
            log_type = "error"
            message = line_strip
        elif "warning" in line_lower:
            log_type = "warning"
            message = line_strip
        elif "note" in line_lower:
            log_type = "info"
            message = line_strip

        if log_type:
            words = re.findall(r'\b[\w_]+\b', message)
            matched_comp = None
            matched_node = None

            # Look for matching component in the message
            for comp in components:
                name = comp.name if hasattr(comp, 'name') else comp.get('name', '')
                if name and any(w.lower() == name.lower() for w in words):
                    matched_comp = name
                    break

            # Look for node name (numeric or vsp_ / vsn_ variables)
            for w in words:
                if w.isdigit() or w.lower().startswith("vsp_") or w.lower().startswith("vsn_") or w.lower() == "vsp" or w.lower() == "vsn":
                    matched_node = w

            netlist_line_num = None
            netlist_line_text = None

            # If we matched a component, search the netlist for that component
            if matched_comp:
                netlist_line_num, netlist_line_text = find_netlist_line(matched_comp, check_contains=False)

            # If no component matched but we have a node, search for the node
            if not netlist_line_num and matched_node:
                netlist_line_num, netlist_line_text = find_netlist_line(matched_node, check_contains=True)

            # Check if warning explicitly mentions a line number
            line_match = re.search(r'\bline\s+(\d+)\b', message, re.I)
            if line_match:
                netlist_line_num = int(line_match.group(1))
                if 1 <= netlist_line_num <= len(netlist_lines):
                    netlist_line_text = netlist_lines[netlist_line_num - 1]

            logs.append({
                "type": log_type,
                "message": message,
                "source": source,
                "component": matched_comp,
                "node": matched_node,
                "line_number": netlist_line_num,
                "line_text": netlist_line_text
            })

    # Scan the netlist for internal error comments generated by template failures
    for idx, line in enumerate(netlist_lines):
        if "* error:" in line.lower() or "* warning:" in line.lower():
            log_type = "error" if "* error:" in line.lower() else "warning"
            msg = line.replace("* ERROR:", "").replace("* WARNING:", "").strip()
            
            matched_comp = None
            for comp in components:
                name = comp.name if hasattr(comp, 'name') else comp.get('name', '')
                if name and name.lower() in msg.lower():
                    matched_comp = name
                    break

            logs.append({
                "type": log_type,
                "message": msg,
                "source": "netlist",
                "component": matched_comp,
                "node": None,
                "line_number": idx + 1,
                "line_text": line
            })

    return logs



def generate_plot_images_matplotlib(
    sim_data: dict,
    plot_config: Dict[str, List[str]],
    colors_config: Dict[str, str],
    sim_dir: str,
    sim_mode: str,
) -> List[str]:
    """
    Fallback: Generate PNG plot images from parsed simulation data using
    matplotlib when ngspice hardcopy/SVG output is unavailable.

    Returns a list of generated filenames.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for server use
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib not available for fallback plot generation")
        return []

    # Only tabular data (list of lists) can be plotted
    if not sim_data.get("values") or not isinstance(sim_data["values"], list):
        return []
    if len(sim_data["values"]) == 0:
        return []
    if not isinstance(sim_data["values"][0], list):
        return []

    vectors = sim_data["vectors"]
    values = sim_data["values"]

    # Convert to columnar format
    num_cols = len(vectors)
    columns: Dict[str, List[float]] = {}
    for i in range(num_cols):
        col_data = []
        for row in values:
            if i < len(row):
                col_data.append(row[i])
        columns[vectors[i]] = col_data

    # x-axis is always the first column (time, frequency, source value)
    x_var = vectors[0]
    x_data = columns[x_var]

    # Color configuration
    bg_color = colors_config.get("0", "white")
    text_color = colors_config.get("1", "black")

    plot_files: List[str] = []
    color_idx = 2

    for win_id in sorted(plot_config.keys()):
        signals = plot_config[win_id]

        fig, ax = plt.subplots(figsize=(9, 5.5))
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        ax.tick_params(colors=text_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.title.set_color(text_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(text_color)

        plotted_any = False
        for sig in signals:
            sig_lower = sig.lower()
            # Find matching column (case-insensitive)
            matched_col = None
            for col_name in vectors:
                if col_name.lower() == sig_lower:
                    matched_col = col_name
                    break

            if matched_col and matched_col in columns:
                sig_color = colors_config.get(str(color_idx), "red")
                ax.plot(
                    x_data, columns[matched_col],
                    color=sig_color, linewidth=2, label=sig
                )
                color_idx += 1
                plotted_any = True

        if not plotted_any:
            plt.close(fig)
            continue

        # Axis labels
        x_labels = {
            "tran": "Time (s)",
            "dc": "Voltage (V)",
            "ac": "Frequency (Hz)",
        }
        ax.set_xlabel(x_labels.get(sim_mode, x_var))
        ax.set_ylabel("Amplitude")
        ax.legend(
            facecolor=bg_color, edgecolor=text_color,
            labelcolor=text_color, framealpha=0.8
        )
        ax.set_title(f"Graph Window {win_id}")
        ax.grid(True, alpha=0.3, color=text_color)

        filename = f"plot_win{win_id}.png"
        filepath = os.path.join(sim_dir, filename)
        fig.savefig(
            filepath, dpi=120, bbox_inches='tight',
            facecolor=fig.get_facecolor(), edgecolor='none'
        )
        plt.close(fig)
        plot_files.append(filename)

    return plot_files


# ═══════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
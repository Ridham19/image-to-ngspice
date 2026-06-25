"""
WebSpice Studio — FastAPI Backend
Endpoints:
  POST /api/detect     — Upload image, run YOLO + OCR, return components
  POST /api/simulate   — Receive canvas state, generate SPICE netlist, run ngspice
"""
import os
import sys
import shutil
import json
import subprocess
import re
from collections import defaultdict
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

# Import ML Logic
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.model import ComponentDetector

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
# NGSPICE CONFIGURATION
# ═══════════════════════════════════════════
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")
NGSPICE_PATH = "ngspice"  # fallback: expect on system PATH

if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, "r") as f:
            cfg = json.load(f)
            NGSPICE_PATH = cfg.get("ngspice_path", NGSPICE_PATH)
        print(f"📋 Loaded ngspice path from config: {NGSPICE_PATH}")
    except Exception as e:
        print(f"⚠️ Could not read config.json: {e}")

# ═══════════════════════════════════════════
# COMPONENT DATABASE (Server-side mirror)
# Matches frontend COMPONENT_DB & PySpice_studio/library.py
# ═══════════════════════════════════════════
GRID_SIZE = 20

PIN_MAP = {
    'resistor':       [(-40, 0), (40, 0)],
    'capacitor':      [(-40, 0), (40, 0)],
    'inductor':       [(-40, 0), (40, 0)],
    'diode':          [(-40, 0), (40, 0)],
    'source':         [(0, -40), (0, 40)],
    'voltage_source': [(0, -40), (0, 40)],
    'current_source': [(0, -40), (0, 40)],
    'ac_source':      [(0, -40), (0, 40)],
    'ground':         [(0, -20)],
    'bjt_npn':        [(-20, 0), (20, -40), (20, 40)],  # Base, Collector, Emitter
    'bjt_pnp':        [(-20, 0), (20, -40), (20, 40)],
    'bjt':            [(-20, 0), (20, -40), (20, 40)],
}

SPICE_TEMPLATES = {
    'resistor':       '{name} {n1} {n2} {value}',
    'capacitor':      '{name} {n1} {n2} {value} ic={ic}',
    'inductor':       '{name} {n1} {n2} {value} ic={ic}',
    'diode':          '{name} {n1} {n2} {model}',
    'source':         '{name} {n1} {n2} DC {dc}',
    'voltage_source': '{name} {n1} {n2} DC {dc}',
    'current_source': '{name} {n1} {n2} DC {dc}',
    'ac_source':      '{name} {n1} {n2} AC {mag} {phase}',
    'bjt_npn':        '{name} {n2} {n1} {n3} {model}',
    'bjt_pnp':        '{name} {n2} {n1} {n3} {model}',
    'bjt':            '{name} {n2} {n1} {n3} {model}',
}

# Types that are not circuit devices (no SPICE line emitted)
NON_DEVICE_TYPES = {'ground', 'label', 'junction', 'wire', 'text'}

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
    mode: str = "op"
    params: Dict[str, str] = {}

class SimulateRequest(BaseModel):
    components: List[ComponentPayload]
    wires: List[List[WirePoint]]
    simConfig: SimConfig = SimConfig()

# ═══════════════════════════════════════════
# LOAD ML MODEL
# ═══════════════════════════════════════════
print("🚀 Booting up API Server...")
detector = ComponentDetector(model_name="../weights/best.pt")

# ═══════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════

@app.get("/")
def read_root():
    return {"status": "Online", "message": "Welcome to the Image-to-SPICE Engine"}


@app.post("/api/detect")
async def detect_circuit(file: UploadFile = File(...)):
    """Upload an image, run YOLO detection + OCR, return detected components."""
    try:
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        json_output_path = os.path.join(WORKSPACE_DIR, "latest_detection.json")
        detected_comps = detector.detect(file_location, output_file=json_output_path)

        # ── Wire Tracing Pipeline ──
        # Step 1: Edge detection + morphological cleanup
        # Step 2: BBOX collision masking (blanks component interiors)
        # Step 3: DFS graph traversal for nodal analysis
        connections = None
        try:
            from core.processing import preprocess_image, separate_layers
            from core.netlist import trace_nodes

            # Step 1 — Preprocessing
            original, gray, binary = preprocess_image(file_location)

            # Step 2 — Collision masking: pass detections so component
            # interiors are blanked out, leaving only external wire traces
            _, wire_mask, _ = separate_layers(gray, binary, detections=detected_comps)

            # Step 3 — DFS traversal from pin anchors through wire pixels
            # This mutates detected_comps in-place (adds "nodes" key)
            connections = trace_nodes(wire_mask, detected_comps)

            print(f"✅ Wire tracing complete: {len(connections or [])} wire segments found")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"⚠️ Wire tracing failed, skipping connections: {e}")

        safe_components = jsonable_encoder(detected_comps)
        safe_connections = jsonable_encoder(connections)

        return JSONResponse(content={
            "status": "success",
            "components": safe_components,
            "connections": safe_connections
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/api/simulate")
async def simulate_circuit(request: SimulateRequest):
    """
    Receive canvas state, generate SPICE netlist via DFS graph traversal,
    write to file, run ngspice, return results.
    """
    try:
        # ─── STEP 1: Compute pin coordinates for all components ───
        comp_pins = []  # List of (component, [pin_coords])
        for comp in request.components:
            pins = compute_pins(comp)
            comp_pins.append((comp, pins))

        # ─── STEP 2: Build coordinate adjacency list ───
        adj = defaultdict(set)

        # Add wire segments to adjacency
        for wire in request.wires:
            if len(wire) >= 2:
                p1 = snap_coord(wire[0].x, wire[0].y)
                p2 = snap_coord(wire[1].x, wire[1].y)
                adj[p1].add(p2)
                adj[p2].add(p1)

        # Add component pin positions as graph nodes
        for comp, pins in comp_pins:
            for pin in pins:
                pt = (pin[0], pin[1])
                if pt not in adj:
                    adj[pt] = set()

        # ─── STEP 3: DFS to cluster connected coordinates into nodes ───
        visited = set()
        node_map = {}  # coordinate -> node_id
        node_counter = 1

        # Find all ground locations
        gnd_coords = set()
        for comp, pins in comp_pins:
            if comp.type == 'ground':
                for pin in pins:
                    gnd_coords.add((pin[0], pin[1]))

        # DFS traversal
        all_points = set(adj.keys())
        for comp, pins in comp_pins:
            for pin in pins:
                all_points.add((pin[0], pin[1]))

        for start_pt in all_points:
            if start_pt in visited:
                continue

            # DFS to find all connected coordinates
            cluster = []
            is_ground = False
            stack = [start_pt]

            while stack:
                curr = stack.pop()
                if curr in visited:
                    continue
                visited.add(curr)
                cluster.append(curr)

                if curr in gnd_coords:
                    is_ground = True

                # Traverse adjacency
                for neighbor in adj.get(curr, set()):
                    if neighbor not in visited:
                        stack.append(neighbor)

            # Assign node ID
            node_id = "0" if is_ground else str(node_counter)
            if not is_ground:
                node_counter += 1

            for pt in cluster:
                node_map[pt] = node_id

        # ─── STEP 4: Generate SPICE netlist ───
        lines = ["* WebSpice Studio — Generated Netlist", ""]

        # Collect required .model statements
        models_needed = set()
        for comp, _ in comp_pins:
            if comp.type == 'diode':
                models_needed.add(".Model Dx diode (Is=14n Rs=0 N=1)")
            elif comp.type in ('bjt_npn', 'bjt'):
                models_needed.add(".Model Tx NPN (BF=300)")
            elif comp.type == 'bjt_pnp':
                models_needed.add(".Model Tx_pnp PNP (BF=300)")

        for m in sorted(models_needed):
            lines.append(m)
        if models_needed:
            lines.append("")

        # Generate device lines
        for comp, pins in comp_pins:
            if comp.type in NON_DEVICE_TYPES:
                continue

            template = SPICE_TEMPLATES.get(comp.type)
            if not template:
                continue

            # Look up node IDs for this component's pins
            nodes = []
            for pin in pins:
                pt = (pin[0], pin[1])
                nodes.append(node_map.get(pt, "NC"))

            # Build context dict for template formatting
            ctx = {
                'name': comp.name,
                'n1': nodes[0] if len(nodes) > 0 else '0',
                'n2': nodes[1] if len(nodes) > 1 else '0',
                'n3': nodes[2] if len(nodes) > 2 else '0',
                'n4': nodes[3] if len(nodes) > 3 else '0',
            }
            # Merge component params
            for k, v in comp.params.items():
                ctx[k] = v

            try:
                line = template.format(**ctx)
                lines.append(line)
            except KeyError as e:
                lines.append(f"* ERROR: Missing param {e} for {comp.name}")

        # ─── STEP 5: Append simulation command ───
        lines.append("")
        sim_mode = request.simConfig.mode.lower()
        sim_params = request.simConfig.params

        if sim_mode == "tran":
            step = sim_params.get("step", "0.1m")
            stop = sim_params.get("stop", "80m")
            start = sim_params.get("start", "0")
            lines.append(f".tran {step} {stop} {start}")
        elif sim_mode == "dc":
            src = sim_params.get("source", "V1")
            start = sim_params.get("start", "0")
            stop = sim_params.get("stop", "5")
            incr = sim_params.get("incr", "0.1")
            lines.append(f".dc {src} {start} {stop} {incr}")
        elif sim_mode == "ac":
            atype = sim_params.get("type", "DEC")
            pts = sim_params.get("points", "10")
            fstart = sim_params.get("fstart", "1")
            fstop = sim_params.get("fstop", "10meg")
            lines.append(f".ac {atype} {pts} {fstart} {fstop}")
        else:
            lines.append(".op")

        # Control block
        lines.append(".control")
        lines.append("run")
        lines.append("set color0 = white")
        lines.append("set color1 = black")
        lines.append("set xbrushwidth = 2")
        lines.append("print all")
        lines.append(".endc")
        lines.append(".end")

        netlist_text = "\n".join(lines)

        # ─── STEP 6: Write netlist to file ───
        cir_path = os.path.join(SIM_DIR, "circuit.cir")
        with open(cir_path, "w", encoding="utf-8") as f:
            f.write(netlist_text)
        print(f"📝 Netlist written to {cir_path}")

        # ─── STEP 7: Run ngspice subprocess ───
        raw_output = ""
        sim_data = {}

        try:
            if not os.path.exists(NGSPICE_PATH) and NGSPICE_PATH != "ngspice":
                return JSONResponse(content={
                    "status": "error",
                    "message": f"ngspice binary not found at: {NGSPICE_PATH}",
                    "netlist": netlist_text,
                    "raw_output": ""
                })

            proc = subprocess.Popen(
                [NGSPICE_PATH, "-b", cir_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=SIM_DIR
            )

            try:
                raw_output, _ = proc.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                raw_output, _ = proc.communicate()
                return JSONResponse(content={
                    "status": "error",
                    "message": "Simulation timed out after 30 seconds.",
                    "netlist": netlist_text,
                    "raw_output": raw_output
                })

            # ─── STEP 8: Parse ngspice output ───
            sim_data = parse_ngspice_output(raw_output)

            if proc.returncode != 0 and not sim_data:
                return JSONResponse(content={
                    "status": "error",
                    "message": f"ngspice exited with code {proc.returncode}",
                    "netlist": netlist_text,
                    "raw_output": raw_output
                })

        except FileNotFoundError:
            return JSONResponse(content={
                "status": "error",
                "message": f"ngspice executable not found. Tried: {NGSPICE_PATH}. Install ngspice or update config.json.",
                "netlist": netlist_text,
                "raw_output": ""
            })

        return JSONResponse(content={
            "status": "success",
            "netlist": netlist_text,
            "raw_output": raw_output,
            "data": sim_data
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": str(e),
            "netlist": "",
            "raw_output": ""
        })


# ═══════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════

def snap_coord(x: float, y: float):
    """Snap a coordinate to the grid and return as a hashable tuple."""
    return (round(x / GRID_SIZE) * GRID_SIZE, round(y / GRID_SIZE) * GRID_SIZE)


def compute_pins(comp: ComponentPayload):
    """
    Compute absolute pin positions for a component based on its type,
    position, and rotation. Returns list of (x, y) tuples snapped to grid.
    """
    pin_offsets = PIN_MAP.get(comp.type, [(-40, 0), (40, 0)])
    rotation_rad = (comp.rotation % 360) * 3.14159265 / 180.0

    import math
    cos_r = round(math.cos(rotation_rad))
    sin_r = round(math.sin(rotation_rad))

    pins = []
    for dx, dy in pin_offsets:
        # Apply rotation
        rx = dx * cos_r - dy * sin_r
        ry = dx * sin_r + dy * cos_r
        # Absolute position, snapped
        px = round((comp.x + rx) / GRID_SIZE) * GRID_SIZE
        py = round((comp.y + ry) / GRID_SIZE) * GRID_SIZE
        pins.append((px, py))

    return pins


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


# ═══════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
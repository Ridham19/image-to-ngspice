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

# SPICE prefix order for sorted netlist output
SPICE_PREFIX_ORDER = {'C': 0, 'D': 1, 'I': 2, 'L': 3, 'Q': 4, 'R': 5, 'V': 6}

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
    write to file, run ngspice, return results + plot images.
    """
    try:
        # ─── STEP 1: Solve node topology via DFS ───
        node_map, comp_pins = solve_canvas(
            request.components, request.wires, GRID_SIZE
        )

        # ─── STEP 2: Generate SPICE netlist ───
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

        # Generate device lines — sorted by SPICE prefix character
        device_lines = []
        for comp, pins in comp_pins:
            if comp.type in NON_DEVICE_TYPES:
                continue

            template = SPICE_TEMPLATES.get(comp.type)
            if not template:
                continue

            # Look up node IDs for this component's pins
            nodes = []
            for pin in pins:
                nodes.append(node_map.get(pin, "NC"))

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
        lines.append("print all")

        # Plot commands — generate hardcopy SVG images for each plot window
        # Clean up old plot files first
        for old_file in glob.glob(os.path.join(SIM_DIR, "plot_win*.svg")):
            try:
                os.remove(old_file)
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

        try:
            if not os.path.exists(NGSPICE_PATH) and NGSPICE_PATH != "ngspice":
                return JSONResponse(content={
                    "status": "error",
                    "message": f"ngspice binary not found at: {NGSPICE_PATH}",
                    "netlist": netlist_text,
                    "raw_output": "",
                    "plot_images": []
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
                    "raw_output": raw_output,
                    "plot_images": []
                })

            # ─── STEP 7: Parse ngspice output ───
            sim_data = parse_ngspice_output(raw_output)

            if proc.returncode != 0 and not sim_data:
                return JSONResponse(content={
                    "status": "error",
                    "message": f"ngspice exited with code {proc.returncode}",
                    "netlist": netlist_text,
                    "raw_output": raw_output,
                    "plot_images": []
                })

        except FileNotFoundError:
            return JSONResponse(content={
                "status": "error",
                "message": f"ngspice executable not found. Tried: {NGSPICE_PATH}. Install ngspice or update config.json.",
                "netlist": netlist_text,
                "raw_output": "",
                "plot_images": []
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
        source_types = {'source', 'voltage_source', 'current_source', 'ac_source'}
        sweepable_types = source_types | {'resistor'}
        for comp, _ in comp_pins:
            if comp.type in source_types:
                source_names.append(comp.name)
            if comp.type in sweepable_types:
                sweepable_names.append(comp.name)

        return JSONResponse(content={
            "status": "success",
            "netlist": netlist_text,
            "raw_output": raw_output,
            "data": sim_data,
            "plot_images": plot_images,
            "nodes": unique_nodes,
            "sources": source_names,
            "sweepables": sweepable_names,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": str(e),
            "netlist": "",
            "raw_output": "",
            "plot_images": []
        })


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

        source_names = []
        sweepable_names = []
        source_types = {'source', 'voltage_source', 'current_source', 'ac_source'}
        sweepable_types = source_types | {'resistor'}
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
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": str(e),
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
"""
core/node_solver.py — Canvas-Side Grid Graph DFS Node Clustering Solver

Implements the three-phase electrical node clustering algorithm for
canvas-originated topologies (both AI-imported and manually placed components).

Pipeline:
  1. Pixel-to-Grid Conversion — snap all coordinates to gridSize multiples
  2. Adjacency Matrix Construction — build structural grid adjacency list
     with full interpolation along multi-cell wire segments
  3. DFS Clustering — identify interconnected terminal groups
  4. Ground Override — relabel ground-connected clusters to node "0"
"""

import math
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Set, Optional


# ═══════════════════════════════════════════════════════════════════════
# PIN MAP — mirrors the server-side PIN_MAP in app.py and the frontend
# COMPONENT_DB pin offsets.  Units are canvas world pixels.
# ═══════════════════════════════════════════════════════════════════════
PIN_MAP: Dict[str, List[Tuple[int, int]]] = {
    'resistor':       [(-40, 0), (40, 0)],
    'capacitor':      [(-40, 0), (40, 0)],
    'inductor':       [(-40, 0), (40, 0)],
    'diode':          [(-40, 0), (40, 0)],
    'source':         [(0, -40), (0, 40)],
    'voltage_source': [(0, -40), (0, 40)],
    'current_source': [(0, -40), (0, 40)],
    'ac_source':      [(0, -40), (0, 40)],
    'pulse_source':   [(0, -40), (0, 40)],
    'sine_source':    [(0, -40), (0, 40)],
    'exp_source':     [(0, -40), (0, 40)],
    'pwl_source':     [(0, -40), (0, 40)],
    'sffm_source':    [(0, -40), (0, 40)],
    'am_source':      [(0, -40), (0, 40)],
    'ground':         [(0, -20)],
    'label':          [(0, 0)],
    'bjt_npn':        [(-20, 0), (20, -40), (20, 40)],
    'bjt_pnp':        [(-20, 0), (20, -40), (20, 40)],
    'bjt':            [(-20, 0), (20, -40), (20, 40)],
}

# Types that don't generate SPICE device lines
NON_DEVICE_TYPES = {'ground', 'label', 'junction', 'wire', 'text'}


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1 — PIXEL-TO-GRID CONVERSION
# ═══════════════════════════════════════════════════════════════════════

def snap_to_grid(x: float, y: float, grid_size: int = 20) -> Tuple[int, int]:
    """Snap a floating-point coordinate to the nearest grid intersection."""
    return (
        round(x / grid_size) * grid_size,
        round(y / grid_size) * grid_size,
    )


def compute_component_pins(
    comp_type: str,
    comp_x: float,
    comp_y: float,
    rotation: int = 0,
    grid_size: int = 20,
) -> List[Tuple[int, int]]:
    """
    Compute absolute grid-snapped pin positions for a component.

    Applies rotation transform to the pin offsets from PIN_MAP,
    then translates to the component's world position and snaps
    to the grid.
    """
    pin_offsets = PIN_MAP.get(comp_type, [(-40, 0), (40, 0)])
    rotation_rad = (rotation % 360) * math.pi / 180.0
    cos_r = round(math.cos(rotation_rad))
    sin_r = round(math.sin(rotation_rad))

    pins: List[Tuple[int, int]] = []
    for dx, dy in pin_offsets:
        # Apply rotation matrix
        rx = dx * cos_r - dy * sin_r
        ry = dx * sin_r + dy * cos_r
        # Absolute position, snapped to grid
        px = round((comp_x + rx) / grid_size) * grid_size
        py = round((comp_y + ry) / grid_size) * grid_size
        pins.append((px, py))

    return pins


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2 — ADJACENCY MATRIX CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════

def build_wire_adjacency(
    wires: List[List[Dict[str, float]]],
    grid_size: int = 20,
) -> Dict[Tuple[int, int], Set[Tuple[int, int]]]:
    """
    Construct a grid-snapped adjacency list from frontend wire segments.

    Each wire is a list of 2+ point dicts: [{x, y}, {x, y}].
    We interpolate along the wire at ``grid_size`` increments so that
    long wire segments spanning multiple grid cells are fully connected
    in the adjacency graph — not just their endpoints.
    """
    adj: Dict[Tuple[int, int], Set[Tuple[int, int]]] = defaultdict(set)

    for wire in wires:
        if len(wire) < 2:
            continue

        # Process all consecutive pairs of points in the wire polyline
        for seg_idx in range(len(wire) - 1):
            pt_a = wire[seg_idx]
            pt_b = wire[seg_idx + 1]

            # Extract coordinates (support both dict and object formats)
            ax = pt_a['x'] if isinstance(pt_a, dict) else pt_a.x
            ay = pt_a['y'] if isinstance(pt_a, dict) else pt_a.y
            bx = pt_b['x'] if isinstance(pt_b, dict) else pt_b.x
            by = pt_b['y'] if isinstance(pt_b, dict) else pt_b.y

            # Snap both endpoints to grid
            p1 = snap_to_grid(ax, ay, grid_size)
            p2 = snap_to_grid(bx, by, grid_size)

            # Interpolate intermediate grid points along the segment
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]

            steps_x = abs(dx) // grid_size if dx != 0 else 0
            steps_y = abs(dy) // grid_size if dy != 0 else 0
            steps = max(steps_x, steps_y)

            if steps == 0:
                # Both endpoints snap to the same grid point — ensure node exists
                if p1 not in adj:
                    adj[p1] = set()
                continue

            # Walk along the wire, adding bidirectional edges between
            # consecutive grid points
            prev = p1
            for i in range(1, steps + 1):
                t = i / steps
                ix = round((p1[0] + dx * t) / grid_size) * grid_size
                iy = round((p1[1] + dy * t) / grid_size) * grid_size
                curr = (ix, iy)

                adj[prev].add(curr)
                adj[curr].add(prev)
                prev = curr

    return adj


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3 — DFS CLUSTERING WITH GROUND OVERRIDE
# ═══════════════════════════════════════════════════════════════════════

def cluster_nodes_dfs(
    comp_pins: List[Tuple[Any, List[Tuple[int, int]]]],
    wire_adj: Dict[Tuple[int, int], Set[Tuple[int, int]]],
    ground_coords: Set[Tuple[int, int]],
    label_coords: Dict[Tuple[int, int], str],
) -> Dict[Tuple[int, int], str]:
    """
    Core DFS clustering algorithm.

    Merges component pin coordinates into the wire adjacency graph,
    then performs iterative DFS from every unvisited node to discover
    connected clusters.  Assigns integer node IDs, with ground-connected
    clusters overridden to the SPICE ground token ``"0"``.

    Parameters
    ----------
    comp_pins     : list of (component_data, [pin_coords])
    wire_adj      : adjacency list from ``build_wire_adjacency``
    ground_coords : set of grid coordinates belonging to ground components

    Returns
    -------
    node_map : dict mapping ``(x, y)`` grid coords → node ID string
    """
    # Deep-copy the wire adjacency so we can mutate it safely
    adj: Dict[Tuple[int, int], Set[Tuple[int, int]]] = defaultdict(set)
    for k, v in wire_adj.items():
        adj[k] = set(v)

    # Inject pin coordinates as graph nodes (isolated if no wire touches them)
    for _, pins in comp_pins:
        for pin in pins:
            if pin not in adj:
                adj[pin] = set()

    # Collect the full set of graph vertices
    all_points: Set[Tuple[int, int]] = set(adj.keys())
    for _, pins in comp_pins:
        for pin in pins:
            all_points.add(pin)

    # Iterative DFS traversal
    visited: Set[Tuple[int, int]] = set()
    node_map: Dict[Tuple[int, int], str] = {}
    node_counter = 1

    for start_pt in all_points:
        if start_pt in visited:
            continue

        cluster: List[Tuple[int, int]] = []
        is_ground = False
        label_text = None
        stack = [start_pt]

        while stack:
            curr = stack.pop()
            if curr in visited:
                continue
            visited.add(curr)
            cluster.append(curr)

            # Check if this coordinate belongs to a ground terminal
            if curr in ground_coords:
                is_ground = True
            
            # Check if this coordinate belongs to a label
            if curr in label_coords:
                label_text = label_coords[curr]

            # Traverse adjacency edges
            for neighbor in adj.get(curr, set()):
                if neighbor not in visited:
                    stack.append(neighbor)

        # Ground override: the SPICE ground token is always "0"
        if is_ground:
            node_id = "0"
        elif label_text is not None:
            node_id = label_text
        else:
            node_id = str(node_counter)
            node_counter += 1

        for pt in cluster:
            node_map[pt] = node_id

    return node_map


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API — solve_canvas
# ═══════════════════════════════════════════════════════════════════════

def solve_canvas(
    components: List[Any],
    wires: List[Any],
    grid_size: int = 20,
) -> Tuple[
    Dict[Tuple[int, int], str],
    List[Tuple[Any, List[Tuple[int, int]]]],
]:
    """
    Top-level orchestrator for the canvas node-clustering solver.

    Chains:  pin computation → adjacency construction → DFS clustering.

    Parameters
    ----------
    components : list of component objects (Pydantic models or dicts)
                 with ``.type``, ``.x``, ``.y``, ``.rotation``
    wires      : list of wire segments, each a list of point objects/dicts
    grid_size  : grid snap resolution (default 20)

    Returns
    -------
    node_map  : dict mapping ``(x, y)`` → node_id string
    comp_pins : list of ``(component, [pin_coords])`` tuples
    """
    # ── Step 1: Compute pin coordinates for all components ──
    comp_pins: List[Tuple[Any, List[Tuple[int, int]]]] = []
    for comp in components:
        # Support both Pydantic models and plain dicts
        if hasattr(comp, 'type'):
            ctype = comp.type
            cx, cy = comp.x, comp.y
            crot = getattr(comp, 'rotation', 0)
        else:
            ctype = comp['type']
            cx, cy = comp['x'], comp['y']
            crot = comp.get('rotation', 0)

        pins = compute_component_pins(ctype, cx, cy, crot, grid_size)
        comp_pins.append((comp, pins))

    # ── Step 2: Build wire adjacency with multi-cell interpolation ──
    wire_dicts: List[List[Dict[str, float]]] = []
    for wire in wires:
        if len(wire) >= 2:
            wire_pts: List[Dict[str, float]] = []
            for pt in wire:
                if hasattr(pt, 'x'):
                    wire_pts.append({'x': pt.x, 'y': pt.y})
                elif isinstance(pt, dict):
                    wire_pts.append(pt)
                else:
                    wire_pts.append({'x': pt[0], 'y': pt[1]})
            wire_dicts.append(wire_pts)

    wire_adj = build_wire_adjacency(wire_dicts, grid_size)

    # ── Step 3: Identify ground and label terminal coordinates ──
    ground_coords: Set[Tuple[int, int]] = set()
    label_coords: Dict[Tuple[int, int], str] = {}
    for comp, pins in comp_pins:
        ctype = comp.type if hasattr(comp, 'type') else comp['type']
        if ctype == 'ground':
            for pin in pins:
                ground_coords.add(pin)
        elif ctype == 'label':
            cparams = comp.params if hasattr(comp, 'params') else comp.get('params', {})
            val = cparams.get('name', 'LBL')
            for pin in pins:
                label_coords[pin] = val

    # ── Step 4: DFS clustering with ground and label override ──
    node_map = cluster_nodes_dfs(comp_pins, wire_adj, ground_coords, label_coords)

    return node_map, comp_pins

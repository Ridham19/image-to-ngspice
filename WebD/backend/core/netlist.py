"""
core/netlist.py — Graph Search Routing Algorithm (Nodal Analysis)
Step 3: Pixel-to-graph conversion, DFS traversal, and node clustering
for the image-based wire-tracing pipeline.
"""

import numpy as np
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Set, Optional

from core.processing import compute_pin_anchors, TRANSPARENT_TYPES
from core.auto_router import snap_pin_to_wire, extract_wire_segments


# ═══════════════════════════════════════════════════════════════════════
# PIXEL → ADJACENCY LIST
# ═══════════════════════════════════════════════════════════════════════

# 8-connected neighbourhood offsets (dx, dy)
_NEIGHBORS_8 = [
    (-1, -1), (0, -1), (1, -1),
    (-1,  0),          (1,  0),
    (-1,  1), (0,  1), (1,  1),
]


def build_adjacency_list(
    wire_mask: np.ndarray,
) -> Dict[Tuple[int, int], Set[Tuple[int, int]]]:
    """
    Convert the binary wire mask into a coordinate adjacency list.

    Every white pixel (value > 0) becomes a node keyed by ``(x, y)``.
    Edges connect each pixel to its 8-connected white neighbours.

    Parameters
    ----------
    wire_mask : np.ndarray
        Single-channel uint8 image where 255 = wire, 0 = background.

    Returns
    -------
    adj : dict[(x, y)] → set[(nx, ny)]
        Adjacency list.  Coordinates use (x, y) = (col, row) convention
        to match OpenCV / image coordinate ordering.
    """
    h, w = wire_mask.shape[:2]

    # Find all white pixel locations — returns (row_array, col_array)
    rows, cols = np.where(wire_mask > 0)

    # Build a fast lookup set for O(1) membership tests
    white_set: Set[Tuple[int, int]] = set()
    for r, c in zip(rows, cols):
        white_set.add((int(c), int(r)))  # (x, y)

    adj: Dict[Tuple[int, int], Set[Tuple[int, int]]] = defaultdict(set)

    for (x, y) in white_set:
        for dx, dy in _NEIGHBORS_8:
            nx, ny = x + dx, y + dy
            if (nx, ny) in white_set:
                adj[(x, y)].add((nx, ny))

    # Ensure isolated white pixels still appear as nodes (empty neighbor set)
    for pt in white_set:
        if pt not in adj:
            adj[pt] = set()

    return adj


# ═══════════════════════════════════════════════════════════════════════
# DFS-BASED NODE CLUSTERING
# ═══════════════════════════════════════════════════════════════════════

def _dfs_collect(
    start: Tuple[int, int],
    adj: Dict[Tuple[int, int], Set[Tuple[int, int]]],
    visited: Set[Tuple[int, int]],
) -> Set[Tuple[int, int]]:
    """
    Iterative DFS starting from *start*, collecting all reachable pixels
    into a cluster set.  Marks every visited pixel in *visited*.
    """
    cluster: Set[Tuple[int, int]] = set()
    stack = [start]

    while stack:
        curr = stack.pop()
        if curr in visited:
            continue
        visited.add(curr)
        cluster.add(curr)

        for neighbor in adj.get(curr, set()):
            if neighbor not in visited:
                stack.append(neighbor)

    return cluster


# ═══════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT — trace_nodes
# ═══════════════════════════════════════════════════════════════════════

def trace_nodes(
    wire_mask: np.ndarray,
    detections: List[Dict[str, Any]],
    snap_radius: int = 15,
) -> List[Dict[str, Any]]:
    """
    Core network solver.  Discovers how YOLO-detected components are
    connected by the hand-drawn wires visible in *wire_mask*.

    Algorithm
    ---------
    1. Compute pin anchors from YOLO bounding boxes.
    2. Build an 8-connected adjacency list from the wire mask.
    3. Snap each pin anchor onto the nearest wire pixel (within *snap_radius*).
    4. DFS from each unvisited (snapped) pin to discover connected clusters.
    5. Additionally DFS from all remaining unvisited wire pixels to capture
       wire segments that don't touch any pin (orphan wires).
    6. Assign node IDs:
       - If a cluster contains a ground pin → node = ``"0"``
       - Otherwise → auto-increment (``"1"``, ``"2"``, …)
    7. Write ``"nodes"`` back into each detection dict in-place.
    8. Package wire clusters into the ``connections`` response list.

    Parameters
    ----------
    wire_mask  : Binary wire-only image (component interiors already masked).
    detections : YOLO detection list (mutated in-place to add ``"nodes"``).
    snap_radius : Max pixel distance for pin-to-wire snapping.

    Returns
    -------
    connections : list of dicts::

        [
            {
                "wire_id": 1,
                "points": [{"x": 120, "y": 240}, {"x": 200, "y": 240}]
            },
            ...
        ]
    """
    # ── 1. Compute pin anchors ──
    pin_anchors = compute_pin_anchors(detections)

    # ── 2. Build pixel adjacency list ──
    adj = build_adjacency_list(wire_mask)

    # ── 3. Snap pins onto the wire graph ──
    # For each pin anchor, find the nearest wire pixel and record the
    # mapping: snapped_coord → list of pin_anchor dicts that map there
    pin_snap_map: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    snapped_pins: List[Optional[Tuple[int, int]]] = []

    for anchor in pin_anchors:
        snapped = snap_pin_to_wire(
            anchor["x"], anchor["y"], wire_mask, radius=snap_radius
        )
        snapped_pins.append(snapped)

        if snapped is not None:
            pin_snap_map[snapped].append(anchor)
            # Ensure the snapped coordinate exists in the adjacency list
            if snapped not in adj:
                adj[snapped] = set()

    # ── 4. Identify ground pin coordinates ──
    ground_coords: Set[Tuple[int, int]] = set()
    for anchor, snapped in zip(pin_anchors, snapped_pins):
        if anchor["comp_type"] == "ground" and snapped is not None:
            ground_coords.add(snapped)

    # ── 5. DFS from all snapped pin anchors ──
    visited: Set[Tuple[int, int]] = set()
    clusters: List[Set[Tuple[int, int]]] = []
    cluster_pin_anchors: List[List[Dict[str, Any]]] = []

    # Seed DFS from every snapped pin location
    all_pin_starts = set(
        s for s in snapped_pins if s is not None
    )

    for start_pt in all_pin_starts:
        if start_pt in visited:
            continue
        cluster = _dfs_collect(start_pt, adj, visited)
        clusters.append(cluster)

        # Collect all pin anchors that belong to this cluster
        pins_in_cluster: List[Dict[str, Any]] = []
        for px in cluster:
            if px in pin_snap_map:
                pins_in_cluster.extend(pin_snap_map[px])
        cluster_pin_anchors.append(pins_in_cluster)

    # Also DFS from remaining unvisited wire pixels (orphan wires)
    all_wire_pixels = set(adj.keys())
    for px in all_wire_pixels:
        if px in visited:
            continue
        cluster = _dfs_collect(px, adj, visited)
        clusters.append(cluster)

        pins_in_cluster = []
        for pt in cluster:
            if pt in pin_snap_map:
                pins_in_cluster.extend(pin_snap_map[pt])
        cluster_pin_anchors.append(pins_in_cluster)

    # ── 6. Assign node IDs ──
    node_counter = 1
    # Map: cluster_index → node_id_string
    cluster_node_ids: List[str] = []

    for i, (cluster, pins) in enumerate(zip(clusters, cluster_pin_anchors)):
        # Check if any pixel in this cluster is a ground coordinate
        is_ground = bool(cluster & ground_coords)

        if is_ground:
            node_id = "0"
        else:
            node_id = str(node_counter)
            node_counter += 1

        cluster_node_ids.append(node_id)

    # ── 7. Write nodes back into detection dicts ──
    # Build per-detection, per-pin node assignment
    # det_nodes[comp_idx] = {pin_id: node_string}
    det_nodes: Dict[int, Dict[int, str]] = defaultdict(dict)

    for cluster_idx, pins in enumerate(cluster_pin_anchors):
        node_id = cluster_node_ids[cluster_idx]
        for anchor in pins:
            comp_idx = anchor["comp_idx"]
            pin_id = anchor["pin_id"]
            det_nodes[comp_idx][pin_id] = node_id

    # Apply to detections
    for idx, det in enumerate(detections):
        comp_type = det.get("type", "")
        if comp_type in TRANSPARENT_TYPES:
            continue

        pin_assignments = det_nodes.get(idx, {})

        # Determine how many pins this component has
        from core.processing import PIN_OFFSET_RULES
        expected_pins = len(PIN_OFFSET_RULES.get(comp_type, [(-1.0, 0.0), (1.0, 0.0)]))

        nodes_list = []
        for pid in range(expected_pins):
            nodes_list.append(pin_assignments.get(pid, "NC"))

        det["nodes"] = nodes_list

    # ── 8. Package connections response ──
    connections: List[Dict[str, Any]] = []
    wire_id = 1

    for cluster_idx, cluster in enumerate(clusters):
        # Only include clusters that actually connect pins
        pins = cluster_pin_anchors[cluster_idx]
        if not cluster:
            continue

        # Extract simplified wire path points
        points = extract_wire_segments(cluster)

        if points:
            connections.append({
                "wire_id": wire_id,
                "points": points,
                "node": cluster_node_ids[cluster_idx],
                "pin_count": len(pins),
            })
            wire_id += 1

    return connections

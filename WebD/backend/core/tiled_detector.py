"""
Tiled Image Splitting, Overlap Computation, Alignment Points, and Global NMS
============================================================================
Provides utilities to slice high-resolution circuit schematics into overlapping
tiles, run component detection per-tile to avoid downscaling degradation on small
components, map coordinates back to global image space, and merge duplicate/chopped
components using Non-Maximum Suppression (NMS) with alignment points.
"""

import math
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np


def compute_tiles(
    width: int,
    height: int,
    grid_cols: int = 2,
    grid_rows: int = 2,
    overlap_ratio: float = 0.20,
) -> List[Dict[str, Any]]:
    """
    Computes tile bounding boxes with an overlap ratio across width and height.

    Parameters:
    -----------
    width: int
        Width of the source image in pixels.
    height: int
        Height of the source image in pixels.
    grid_cols: int
        Number of column divisions (default 2).
    grid_rows: int
        Number of row divisions (default 2).
    overlap_ratio: float
        Overlap fraction between adjacent tiles (e.g., 0.20 for 20% overlap).

    Returns:
    --------
    List of tile dictionaries containing:
        - tile_id: int
        - row: int
        - col: int
        - x1, y1, x2, y2: int coordinates in global image space
        - width, height: int dimensions of the tile
        - offset: [x1, y1] for local-to-global coordinate translation
    """
    grid_cols = max(1, int(grid_cols))
    grid_rows = max(1, int(grid_rows))
    overlap_ratio = max(0.0, min(0.50, float(overlap_ratio)))

    if grid_cols == 1 and grid_rows == 1:
        return [{
            "tile_id": 0,
            "row": 0,
            "col": 0,
            "x1": 0,
            "y1": 0,
            "x2": width,
            "y2": height,
            "width": width,
            "height": height,
            "offset": [0, 0],
        }]

    # Calculate tile dimensions tw and th using overlap formula:
    # W = tw + (cols - 1) * tw * (1 - overlap_ratio)
    # tw = W / (1 + (cols - 1) * (1 - overlap_ratio))
    denom_x = 1.0 + (grid_cols - 1) * (1.0 - overlap_ratio) if grid_cols > 1 else 1.0
    tw = width / denom_x
    dx = tw * (1.0 - overlap_ratio) if grid_cols > 1 else width

    denom_y = 1.0 + (grid_rows - 1) * (1.0 - overlap_ratio) if grid_rows > 1 else 1.0
    th = height / denom_y
    dy = th * (1.0 - overlap_ratio) if grid_rows > 1 else height

    tiles = []
    tile_id = 0

    for r in range(grid_rows):
        for c in range(grid_cols):
            x1 = int(round(c * dx))
            y1 = int(round(r * dy))

            if c == grid_cols - 1:
                x2 = width
            else:
                x2 = min(width, int(round(x1 + tw)))

            if r == grid_rows - 1:
                y2 = height
            else:
                y2 = min(height, int(round(y1 + th)))

            # Guarantee non-negative / valid bounding box
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, max(x1 + 1, x2))
            y2 = min(height, max(y1 + 1, y2))

            tiles.append({
                "tile_id": tile_id,
                "row": r,
                "col": c,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": x2 - x1,
                "height": y2 - y1,
                "offset": [x1, y1],
            })
            tile_id += 1

    return tiles


def compute_overlap_bands(
    tiles: List[Dict[str, Any]],
    grid_cols: int,
    grid_rows: int,
    width: int,
    height: int,
) -> Dict[str, Any]:
    """
    Identifies the exact overlapping rectangular regions between adjacent tiles.
    """
    vertical_bands = []
    horizontal_bands = []
    intersections = []

    # Map (row, col) to tile
    tile_map = {(t["row"], t["col"]): t for t in tiles}

    # Vertical overlap strips (between column c and c + 1)
    for c in range(grid_cols - 1):
        left_t = tile_map.get((0, c))
        right_t = tile_map.get((0, c + 1))
        if left_t and right_t:
            overlap_x1 = right_t["x1"]
            overlap_x2 = left_t["x2"]
            if overlap_x2 > overlap_x1:
                vertical_bands.append({
                    "col_pair": [c, c + 1],
                    "x1": overlap_x1,
                    "x2": overlap_x2,
                    "y1": 0,
                    "y2": height,
                    "width": overlap_x2 - overlap_x1,
                })

    # Horizontal overlap strips (between row r and r + 1)
    for r in range(grid_rows - 1):
        top_t = tile_map.get((r, 0))
        bottom_t = tile_map.get((r + 1, 0))
        if top_t and bottom_t:
            overlap_y1 = bottom_t["y1"]
            overlap_y2 = top_t["y2"]
            if overlap_y2 > overlap_y1:
                horizontal_bands.append({
                    "row_pair": [r, r + 1],
                    "x1": 0,
                    "x2": width,
                    "y1": overlap_y1,
                    "y2": overlap_y2,
                    "height": overlap_y2 - overlap_y1,
                })

    # Intersections where vertical and horizontal overlap bands cross (4-tile overlaps)
    for v_band in vertical_bands:
        for h_band in horizontal_bands:
            intersections.append({
                "x1": v_band["x1"],
                "y1": h_band["y1"],
                "x2": v_band["x2"],
                "y2": h_band["y2"],
                "width": v_band["width"],
                "height": h_band["height"],
            })

    return {
        "vertical_bands": vertical_bands,
        "horizontal_bands": horizontal_bands,
        "intersections": intersections,
    }


def compute_alignment_points(
    tiles: List[Dict[str, Any]],
    grid_cols: int,
    grid_rows: int,
    width: int,
    height: int,
) -> List[Dict[str, Any]]:
    """
    Computes key alignment reference points along the overlap seams and intersections.
    These alignment points act as visual and mathematical anchors for stitching.
    """
    bands = compute_overlap_bands(tiles, grid_cols, grid_rows, width, height)
    alignment_points = []
    pt_id = 1

    # 1. Intersection center anchors (where horizontal and vertical overlap zones meet)
    for inter in bands["intersections"]:
        cx = int(round((inter["x1"] + inter["x2"]) / 2.0))
        cy = int(round((inter["y1"] + inter["y2"]) / 2.0))
        alignment_points.append({
            "id": f"align_cross_{pt_id}",
            "type": "intersection",
            "x": cx,
            "y": cy,
            "label": f"AP{pt_id}",
        })
        pt_id += 1

    # 2. Seam boundary midpoints for vertical seams
    for v_band in bands["vertical_bands"]:
        cx = int(round((v_band["x1"] + v_band["x2"]) / 2.0))
        # Top margin, middle, bottom margin
        for frac in (0.15, 0.50, 0.85):
            cy = int(round(height * frac))
            # Avoid duplicating an intersection point
            if not any(abs(p["x"] - cx) < 20 and abs(p["y"] - cy) < 20 for p in alignment_points):
                alignment_points.append({
                    "id": f"align_v_{pt_id}",
                    "type": "vertical_seam",
                    "x": cx,
                    "y": cy,
                    "label": f"AP{pt_id}",
                })
                pt_id += 1

    # 3. Seam boundary midpoints for horizontal seams
    for h_band in bands["horizontal_bands"]:
        cy = int(round((h_band["y1"] + h_band["y2"]) / 2.0))
        for frac in (0.15, 0.50, 0.85):
            cx = int(round(width * frac))
            if not any(abs(p["x"] - cx) < 20 and abs(p["y"] - cy) < 20 for p in alignment_points):
                alignment_points.append({
                    "id": f"align_h_{pt_id}",
                    "type": "horizontal_seam",
                    "x": cx,
                    "y": cy,
                    "label": f"AP{pt_id}",
                })
                pt_id += 1

    return alignment_points


def calculate_box_metrics(box_a: List[int], box_b: List[int]) -> Dict[str, Any]:
    """
    Computes comprehensive geometric overlap and proximity metrics between two bounding boxes [x, y, w, h].
    Includes Intersection over Union (IoU), Intersection over Smaller Box (IoS / Containment),
    Euclidean center distance, and axial offsets.
    """
    ax1, ay1, aw, ah = box_a
    ax2, ay2 = ax1 + aw, ay1 + ah

    bx1, by1, bw, bh = box_b
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x1 - inter_x2 if inter_x1 > inter_x2 else inter_x2 - inter_x1)
    inter_h = max(0, inter_y1 - inter_y2 if inter_y1 > inter_y2 else inter_y2 - inter_y1)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        inter_area = 0
    else:
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    area_a = max(1, aw * ah)
    area_b = max(1, bw * bh)
    union_area = area_a + area_b - inter_area

    iou = (inter_area / float(union_area)) if union_area > 0 else 0.0
    min_area = min(area_a, area_b)
    ios = (inter_area / float(min_area)) if min_area > 0 else 0.0

    ca_x = ax1 + aw / 2.0
    ca_y = ay1 + ah / 2.0
    cb_x = bx1 + bw / 2.0
    cb_y = by1 + bh / 2.0
    dx = abs(ca_x - cb_x)
    dy = abs(ca_y - cb_y)
    dist = math.hypot(dx, dy)

    # Point containment
    a_in_b = (bx1 <= ca_x <= bx2) and (by1 <= ca_y <= by2)
    b_in_a = (ax1 <= cb_x <= ax2) and (ay1 <= cb_y <= ay2)

    return {
        "inter_area": inter_area,
        "iou": iou,
        "ios": ios,
        "dist": dist,
        "dx": dx,
        "dy": dy,
        "a_in_b": a_in_b,
        "b_in_a": b_in_a,
        "max_w": max(aw, bw),
        "max_h": max(ah, bh),
        "min_w": min(aw, bw),
        "min_h": min(ah, bh),
    }


def calculate_iou(box_a: List[int], box_b: List[int]) -> float:
    """Computes Intersection over Union (IoU) for two bounding boxes [x, y, w, h]."""
    return calculate_box_metrics(box_a, box_b)["iou"]


def calculate_center_distance(box_a: List[int], box_b: List[int]) -> float:
    """Computes Euclidean distance between centers of two boxes [x, y, w, h]."""
    return calculate_box_metrics(box_a, box_b)["dist"]


def are_classes_compatible(cls_a: str, cls_b: str) -> bool:
    """
    Checks if two class names represent the same component family.
    e.g. resistor and resistor_photo, bjt and bjt_npn, mosfet and nmos are compatible.
    """
    if cls_a == cls_b:
        return True

    families = [
        {'resistor', 'resistor_photo'},
        {'capacitor', 'capacitor_polarized'},
        {'inductor', 'transformer'},
        {'diode', 'diode_led', 'diode_zener'},
        {'source', 'voltage_source', 'current_source', 'ac_source', 'sine_source',
         'pulse_source', 'exp_source', 'am_source', 'sffm_source', 'pwl_source', 'vss'},
        {'ground', 'vss'},
        {'bjt', 'bjt_npn', 'bjt_pnp', 'transistor', 'phototransistor'},
        {'mosfet', 'nmos', 'pmos', 'transistor'},
        {'opamp', 'ic'},
        {'and_gate', 'or_gate', 'not_gate', 'nand_gate', 'nor_gate', 'xor_gate', 'xnor_gate', 'ic'},
    ]
    for fam in families:
        if cls_a in fam and cls_b in fam:
            return True
    return False


def is_same_component(det_a: Dict[str, Any], det_b: Dict[str, Any]) -> bool:
    """
    Determines whether two detections from (potentially different) tiles represent
    the same physical component in the schematic drawing.
    Uses multi-criteria geometric testing including IoU, IoS (containment), center point
    containment, and axial proximity to robustly merge full and clipped detections.
    """
    box_a = det_a.get("box", [0, 0, 0, 0])
    box_b = det_b.get("box", [0, 0, 0, 0])
    cls_a = det_a.get("type", "")
    cls_b = det_b.get("type", "")

    m = calculate_box_metrics(box_a, box_b)
    comp_classes = are_classes_compatible(cls_a, cls_b)

    # 1. Direct bounding box overlap (IoU)
    # If classes are compatible, 0.15 is enough; if cross-class conflict, require 0.25
    if m["iou"] >= 0.15 and (comp_classes or m["iou"] >= 0.25):
        return True

    # 2. Containment / clipped boundary fragment (IoS)
    # When a component is clipped at a tile seam, one tile sees a smaller slice.
    # IoS measures how much of the smaller box is covered by the intersection.
    if m["ios"] >= 0.35 and (comp_classes or m["ios"] >= 0.50):
        return True

    # 3. Center point containment with positive intersection
    # Center of Box A is inside Box B, or center of Box B is inside Box A
    if m["inter_area"] > 0 and (m["a_in_b"] or m["b_in_a"]):
        return True

    # 4. Normalized center offset within component footprint
    if m["inter_area"] > 0 and m["dx"] <= m["max_w"] * 0.65 and m["dy"] <= m["max_h"] * 0.65:
        return True

    # 5. Proximity within small Euclidean radius
    max_dim = max(m["max_w"], m["max_h"])
    dist_thresh = max(30.0, 0.45 * max_dim)
    if m["dist"] <= dist_thresh:
        # If there is any bounding box contact or the centers are very close (<= 25px)
        if (m["inter_area"] > 0 or m["dist"] <= 25.0) and (comp_classes or m["dist"] <= 20.0):
            return True

    return False


def merge_tiled_detections(
    raw_detections: List[Dict[str, Any]],
    iou_thresh: float = 0.20,
    center_dist_ratio: float = 0.55,
) -> List[Dict[str, Any]]:
    """
    Merges duplicate component detections originating from overlapping tile regions.
    Uses Union-Find connected component clustering over all detections.
    Prioritizes complete, un-chopped detections and preserves parsed OCR values.

    Parameters:
    -----------
    raw_detections: List of detection dicts with global coordinates:
        - type: str
        - box: [x, y, w, h]
        - center: [cx, cy]
        - conf: float
        - value: str or None
        - rotation: int
        - tile_id: int
        - is_near_tile_edge: bool

    Returns:
    --------
    List of merged, unique global detections.
    """
    if not raw_detections:
        return []

    n = len(raw_detections)
    parent = list(range(n))

    def find(i: int) -> int:
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
        return i

    def union(i: int, j: int) -> None:
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_j] = root_i

    # Connect all pairs that represent the same component
    for i in range(n):
        for j in range(i + 1, n):
            if is_same_component(raw_detections[i], raw_detections[j]):
                union(i, j)

    # Group detections by cluster root
    clusters_map = defaultdict(list)
    for i in range(n):
        clusters_map[find(i)].append(raw_detections[i])

    merged_results = []
    prefix_counts = {}

    # SPICE prefix mapping
    prefix_map = {
        'resistor': 'R', 'resistor_photo': 'R',
        'capacitor': 'C', 'capacitor_polarized': 'C',
        'inductor': 'L',
        'diode': 'D', 'diode_led': 'D', 'diode_zener': 'D',
        'source': 'V', 'voltage_source': 'V', 'ac_source': 'V',
        'current_source': 'I',
        'ground': 'GND', 'vss': 'VSS',
        'bjt': 'Q', 'bjt_npn': 'Q', 'bjt_pnp': 'Q',
        'mosfet': 'M', 'nmos': 'M', 'pmos': 'M', 'phototransistor': 'Q',
        'opamp': 'U', 'ic': 'U',
        'transformer': 'T',
        'and_gate': 'U', 'or_gate': 'U', 'not_gate': 'U', 'nand_gate': 'U',
        'nor_gate': 'U', 'xor_gate': 'U', 'xnor_gate': 'U',
        'junction': 'J', 'crossover': 'X', 'terminal': 'P',
    }

    for cluster in clusters_map.values():
        if len(cluster) == 1:
            det = dict(cluster[0])
            merged_results.append(det)
            continue

        # Separate full detections from edge-clipped seam detections
        full_members = [d for d in cluster if not d.get("is_near_tile_edge", False)]
        candidates = full_members if full_members else cluster

        # Primary detection: highest confidence among candidates
        best_det = max(candidates, key=lambda d: d.get("conf", 0.0))

        # Best OCR value across the cluster
        best_val = None
        for d in candidates:
            v = d.get("value")
            if v and v != "None" and str(v).strip() and str(v).strip() != "TEXT_FOUND":
                best_val = v
                break
        if not best_val:
            for d in cluster:
                v = d.get("value")
                if v and v != "None" and str(v).strip() and str(v).strip() != "TEXT_FOUND":
                    best_val = v
                    break

        # Refined specific component type (e.g. bjt_npn over bjt, nmos over mosfet)
        refined_type = best_det.get("type", "")
        specific_types = (
            "bjt_npn", "bjt_pnp", "diode_led", "diode_zener",
            "nmos", "pmos", "resistor_photo", "capacitor_polarized"
        )
        for d in cluster:
            t = d.get("type", "")
            if t in specific_types:
                refined_type = t
                break

        # Bounding box determination:
        # If complete, non-edge detections exist, use the best intact detection box so
        # clipped seam fragments do NOT shrink or distort the component.
        # If all detections were clipped at seams, compute the union envelope to reconstruct the symbol.
        if full_members:
            best_box_det = max(full_members, key=lambda d: (d.get("conf", 0.0), d["box"][2] * d["box"][3]))
            merged_box = list(best_box_det["box"])
        else:
            min_x = min(d["box"][0] for d in cluster)
            min_y = min(d["box"][1] for d in cluster)
            max_x2 = max(d["box"][0] + d["box"][2] for d in cluster)
            max_y2 = max(d["box"][1] + d["box"][3] for d in cluster)
            merged_box = [min_x, min_y, max(1, max_x2 - min_x), max(1, max_y2 - min_y)]

        merged_center = [
            int(round(merged_box[0] + merged_box[2] / 2.0)),
            int(round(merged_box[1] + merged_box[3] / 2.0))
        ]

        merged_det = {
            "type": refined_type,
            "box": merged_box,
            "center": merged_center,
            "conf": round(float(best_det.get("conf", 0.9)), 3),
            "value": best_val,
            "raw_ocr": best_det.get("raw_ocr"),
            "ocr_status": best_det.get("ocr_status", "detected" if best_val else "unset"),
            "text_box": best_det.get("text_box"),
            "rotation": best_det.get("rotation", 0),
            "merged_count": len(cluster),
        }
        merged_results.append(merged_det)

    # Assign sequential schematic designators (R1, R2, C1, etc.)
    for det in merged_results:
        ctype = det.get("type", "resistor")
        prefix = prefix_map.get(ctype, ctype[0].upper() if ctype else 'U')
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
        det["name"] = f"{prefix}{prefix_counts[prefix]}"

    return merged_results


def deduplicate_global_ocr_texts(ocr_texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate OCR text boxes originating from overlapping tile boundaries."""
    deduped: List[Dict[str, Any]] = []
    for ot in ocr_texts:
        b = ot.get("box", [0, 0, 0, 0])
        txt = (ot.get("clean") or ot.get("text") or "").strip().lower()
        is_dup = False
        for ex in deduped:
            eb = ex.get("box", [0, 0, 0, 0])
            etxt = (ex.get("clean") or ex.get("text") or "").strip().lower()
            r1_x1, r1_y1, r1_x2, r1_y2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
            r2_x1, r2_y1, r2_x2, r2_y2 = eb[0], eb[1], eb[0] + eb[2], eb[1] + eb[3]
            dx = max(0.0, float(max(r1_x1 - r2_x2, r2_x1 - r1_x2)))
            dy = max(0.0, float(max(r1_y1 - r2_y2, r2_y1 - r1_y2)))
            edist = math.hypot(dx, dy)
            if edist < 25 and (txt == etxt or not txt or not etxt):
                is_dup = True
                if ot.get("conf", 0) > ex.get("conf", 0):
                    ex["text"] = ot.get("text") or ex.get("text")
                    ex["clean"] = ot.get("clean") or ex.get("clean")
                    ex["conf"] = ot.get("conf", ex.get("conf"))
                break
        if not is_dup:
            ot_copy = dict(ot)
            ot_copy["id"] = f"T{len(deduped) + 1}"
            deduped.append(ot_copy)
    return deduped


def associate_global_ocr_to_components(components: List[Dict[str, Any]], ocr_texts: List[Dict[str, Any]]):
    """Automatically associate nearby global OCR text boxes to merged circuit components."""
    try:
        from core.model import clean_ocr_string, classify_text, correct_component_value
    except ImportError:
        return

    NON_VAL = {'wire', 'junction', 'crossover', 'terminal', 'ground', 'vss'}
    PREFIX_MAP = {
        'resistor': 'R', 'resistor_photo': 'R', 'capacitor': 'C', 'capacitor_polarized': 'C',
        'inductor': 'L', 'source': 'V', 'voltage_source': 'V', 'ac_source': 'V', 'current_source': 'I',
        'diode': 'D', 'diode_led': 'D', 'diode_zener': 'D', 'bjt': 'Q', 'bjt_npn': 'Q',
        'bjt_pnp': 'Q', 'mosfet': 'M', 'nmos': 'M', 'pmos': 'M', 'opamp': 'U'
    }

    def calc_edge_dist(b1, b2):
        if not b1 or not b2 or len(b1) < 4 or len(b2) < 4:
            return 999999.0
        dx = max(0.0, float(max(b1[0] - (b2[0] + b2[2]), b2[0] - (b1[0] + b1[2]))))
        dy = max(0.0, float(max(b1[1] - (b2[1] + b2[3]), b2[1] - (b1[1] + b1[3]))))
        return math.hypot(dx, dy)

    for comp in components:
        if comp.get('type') in NON_VAL:
            continue
        has_val = comp.get('value') and comp.get('value') != 'TEXT_FOUND' and str(comp.get('value')).strip() != ''
        if has_val:
            continue

        comp_box = comp.get('box', [0, 0, 0, 0])
        cw = comp_box[2] if len(comp_box) > 2 else 50
        ch = comp_box[3] if len(comp_box) > 3 else 50
        comp_prefix = PREFIX_MAP.get(comp.get('type', ''), '')
        max_dist = max(260.0, max(cw, ch) * 2.2)

        candidates = []
        for ot in ocr_texts:
            ot_box = ot.get('box', [0, 0, 0, 0])
            edist = calc_edge_dist(comp_box, ot_box)
            if edist > max_dist:
                continue

            raw_txt = ot.get('clean') or ot.get('text') or ''
            cleaned_str = clean_ocr_string(raw_txt)
            classification = classify_text(cleaned_str, comp.get('type', ''), comp_prefix)

            # Check if this text box is significantly closer to another valid component
            closer_to_other = False
            for other_comp in components:
                if other_comp is comp or other_comp.get('type') in NON_VAL:
                    continue
                other_dist = calc_edge_dist(other_comp.get('box', [0, 0, 0, 0]), ot_box)
                if other_dist < edist * 0.75:
                    closer_to_other = True
                    break
            if closer_to_other:
                continue

            val = None
            detected_raw = raw_txt
            if isinstance(classification, dict):
                val = correct_component_value(classification['value'], comp.get('type', ''))
                detected_raw = classification['value']
            elif classification == 'value':
                val = correct_component_value(cleaned_str, comp.get('type', ''))
                detected_raw = cleaned_str

            if val:
                candidates.append((val, detected_raw, ot, edist))

        if candidates:
            candidates.sort(key=lambda x: x[3])
            best_val, best_raw, best_ot, _ = candidates[0]
            comp['value'] = best_val
            comp['raw_ocr'] = best_raw
            comp['ocr_status'] = 'detected'
            comp['text_box'] = best_ot.get('box')
            best_ot['associated_comp'] = comp.get('name')


def split_and_detect_circuit(
    detector,
    bgr_image: np.ndarray,
    grid_cols: int = 2,
    grid_rows: int = 2,
    overlap_ratio: float = 0.20,
    canny_low: int = 50,
    canny_high: int = 150,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Executes the complete tiled detection pipeline on a high-resolution circuit image.

    Steps:
    ------
    1. Divides the image into an overlapping grid.
    2. Runs YOLO + OCR on each individual tile at full resolution.
    3. Transforms bounding boxes and pin locations to global image coordinates.
    4. Merges duplicate component detections across seams using IoU NMS.
    5. Computes alignment points and overlap metadata for visualization.

    Returns:
    --------
    (merged_components, tile_metadata)
    """
    height, width = bgr_image.shape[:2]

    # Compute tile layout, overlap bands, and alignment points
    tiles = compute_tiles(width, height, grid_cols=grid_cols, grid_rows=grid_rows, overlap_ratio=overlap_ratio)
    alignment_points = compute_alignment_points(tiles, grid_cols, grid_rows, width, height)
    overlap_bands = compute_overlap_bands(tiles, grid_cols, grid_rows, width, height)

    all_raw_detections: List[Dict[str, Any]] = []
    all_global_ocr_texts: List[Dict[str, Any]] = []

    print(f"[TILED] Processing image in Tiled Mode: {width}x{height} px -> {grid_cols}x{grid_rows} grid ({len(tiles)} tiles, {int(overlap_ratio * 100)}% overlap)")

    for tile in tiles:
        tx1, ty1, tx2, ty2 = tile["x1"], tile["y1"], tile["x2"], tile["y2"]
        tile_crop = bgr_image[ty1:ty2, tx1:tx2]

        if tile_crop.size == 0:
            continue

        # Run component detector on this crop
        tile_dets = detector.detect(tile_crop)

        # Collect and translate local OCR text detections to global coordinate space
        for ocr_t in getattr(detector, 'latest_ocr_texts', []):
            bx, by, bw, bh = ocr_t.get("box", [0, 0, 0, 0])
            all_global_ocr_texts.append({
                "id": f"T{len(all_global_ocr_texts) + 1}",
                "box": [bx + tx1, by + ty1, bw, bh],
                "text": ocr_t.get("text", ""),
                "clean": ocr_t.get("clean", ""),
                "conf": ocr_t.get("conf", 0.5),
                "associated_comp": None,
            })

        # Map local bounding box coordinates to global coordinates
        edge_margin = 12  # px threshold to flag detections clipped at tile borders
        for det in tile_dets:
            lx, ly, lw, lh = det.get("box", [0, 0, 0, 0])
            gx = lx + tx1
            gy = ly + ty1

            # Check if detection touches or is sliced by an internal tile seam
            is_edge = (
                (tile["x1"] > 0 and lx <= edge_margin) or
                (tile["y1"] > 0 and ly <= edge_margin) or
                (tile["x2"] < width and (lx + lw) >= (tile["width"] - edge_margin)) or
                (tile["y2"] < height and (ly + lh) >= (tile["height"] - edge_margin))
            )

            text_box = None
            if det.get("text_box"):
                tbx, tby, tbw, tbh = det["text_box"]
                text_box = [tbx + tx1, tby + ty1, tbw, tbh]

            global_det = {
                "type": det.get("type", "resistor"),
                "box": [gx, gy, lw, lh],
                "center": [gx + lw // 2, gy + lh // 2],
                "conf": det.get("conf", 0.9),
                "value": det.get("value"),
                "raw_ocr": det.get("raw_ocr"),
                "ocr_status": det.get("ocr_status", "detected" if det.get("value") else "unset"),
                "text_box": text_box,
                "rotation": det.get("rotation", 0),
                "tile_id": tile["tile_id"],
                "is_near_tile_edge": is_edge,
            }
            all_raw_detections.append(global_det)

    # Perform IoU-based deduplication and edge suppression across all tiles
    merged_components = merge_tiled_detections(all_raw_detections, iou_thresh=0.30, center_dist_ratio=0.55)

    # Deduplicate OCR text boxes from overlapping tile boundaries and auto-associate values
    deduped_ocr_texts = deduplicate_global_ocr_texts(all_global_ocr_texts)
    associate_global_ocr_to_components(merged_components, deduped_ocr_texts)
    detector.latest_ocr_texts = deduped_ocr_texts

    tile_metadata = {
        "grid_cols": grid_cols,
        "grid_rows": grid_rows,
        "overlap_ratio": overlap_ratio,
        "image_width": width,
        "image_height": height,
        "num_tiles": len(tiles),
        "tiles": tiles,
        "alignment_points": alignment_points,
        "overlap_bands": overlap_bands,
        "raw_detection_count": len(all_raw_detections),
        "merged_detection_count": len(merged_components),
    }

    print(f"[TILED] Tiled Detection complete: {len(all_raw_detections)} raw tile detections merged into {len(merged_components)} unique circuit components ({len(deduped_ocr_texts)} global OCR texts).")

    return merged_components, tile_metadata


def detect_seam_wire_anchors(
    bgr_image: np.ndarray,
    tiles: List[Dict[str, Any]],
    overlap_bands: Dict[str, Any],
    components: List[Dict[str, Any]],
    canny_low: int = 50,
    canny_high: int = 150,
) -> List[Dict[str, Any]]:
    """
    Detects wire crossing points along the overlapping boundaries between adjacent tiles.
    Places anchor points on the wires along the overlap seams so that adjacent tiles
    share identical anchor coordinates across both sub-images.
    """
    height, width = bgr_image.shape[:2]

    # Preprocess bgr_image to obtain a clean binary wire mask with noise filtering
    from wire_tracer.preprocess import preprocess_image
    wire_mask = preprocess_image(
        bgr_image,
        canny_low=canny_low,
        canny_high=canny_high,
        min_blob_area=40,
    )

    # Mask out component bounding boxes completely to prevent component bodies from generating false seam anchors
    for c in components:
        box = c.get("box", [0, 0, 0, 0])
        if box and len(box) == 4 and box[2] > 4 and box[3] > 4:
            bx, by, bw, bh = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(wire_mask, (bx, by), (bx + bw, by + bh), 0, -1)

    tile_map = {(t["row"], t["col"]): t for t in tiles}
    seam_anchors: List[Dict[str, Any]] = []
    anchor_id = 1
    grid_rows = max(t["row"] for t in tiles) + 1
    grid_cols = max(t["col"] for t in tiles) + 1

    def _create_anchor(anc_type: str, gx: int, gy: int, base_tiles: List[int]) -> Dict[str, Any]:
        nonlocal anchor_id
        matched_tiles = list(base_tiles)
        coords = {}
        for t in tiles:
            tid = t["tile_id"]
            if (t["x1"] - 4 <= gx <= t["x2"] + 4) and (t["y1"] - 4 <= gy <= t["y2"] + 4):
                if tid not in matched_tiles:
                    matched_tiles.append(tid)
        for tid in matched_tiles:
            t = next(it for it in tiles if it["tile_id"] == tid)
            coords[tid] = [int(gx - t["x1"]), int(gy - t["y1"])]

        anc = {
            "id": f"seam_anchor_{anchor_id}",
            "anchor_id": anchor_id,
            "type": anc_type,
            "global_x": int(gx),
            "global_y": int(gy),
            "tiles": matched_tiles,
            "tile_coords": coords,
        }
        anchor_id += 1
        return anc

    # 1. Vertical overlap bands (seams between adjacent columns c and c+1)
    for v_band in overlap_bands.get("vertical_bands", []):
        c_left, c_right = v_band["col_pair"]
        x_seam = int(round((v_band["x1"] + v_band["x2"]) / 2.0))

        for r in range(grid_rows):
            t_left = tile_map.get((r, c_left))
            t_right = tile_map.get((r, c_right))
            if not t_left or not t_right:
                continue

            y_start = max(t_left["y1"], t_right["y1"])
            y_end = min(t_left["y2"], t_right["y2"])
            if y_end <= y_start + 10:
                continue

            # Sample a narrow strip around x_seam
            x_a = max(0, x_seam - 4)
            x_b = min(width, x_seam + 5)
            strip = wire_mask[y_start:y_end, x_a:x_b]
            if strip.size == 0:
                continue

            row_profile = np.max(strip, axis=1)

            # Find contiguous runs of wire pixels along the column
            in_run = False
            start_y = 0
            for y_rel in range(len(row_profile)):
                if row_profile[y_rel] > 0 and not in_run:
                    in_run = True
                    start_y = y_rel
                elif row_profile[y_rel] == 0 and in_run:
                    in_run = False
                    y_mid = y_start + (start_y + y_rel - 1) // 2
                    local_y = (start_y + y_rel - 1) // 2
                    wire_x_offsets = np.where(strip[local_y, :] > 0)[0]
                    best_x = x_seam if len(wire_x_offsets) == 0 else x_a + int(np.mean(wire_x_offsets))

                    # Bilateral continuity check: wire must genuinely extend across both sides of the seam
                    y_w_min = max(0, y_mid - 6)
                    y_w_max = min(height, y_mid + 7)
                    left_ok = np.any(wire_mask[y_w_min:y_w_max, max(0, best_x - 12):best_x] > 0)
                    right_ok = np.any(wire_mask[y_w_min:y_w_max, best_x:min(width, best_x + 13)] > 0)

                    if left_ok and right_ok:
                        if not any(abs(a["global_y"] - y_mid) < 25 for a in seam_anchors if a["type"] == "vertical" and abs(a["global_x"] - best_x) < 25):
                            seam_anchors.append(_create_anchor("vertical", best_x, y_mid, [t_left["tile_id"], t_right["tile_id"]]))

            if in_run:
                y_mid = y_start + (start_y + len(row_profile) - 1) // 2
                local_y = min(len(row_profile) - 1, (start_y + len(row_profile) - 1) // 2)
                wire_x_offsets = np.where(strip[local_y, :] > 0)[0]
                best_x = x_seam if len(wire_x_offsets) == 0 else x_a + int(np.mean(wire_x_offsets))

                y_w_min = max(0, y_mid - 6)
                y_w_max = min(height, y_mid + 7)
                left_ok = np.any(wire_mask[y_w_min:y_w_max, max(0, best_x - 12):best_x] > 0)
                right_ok = np.any(wire_mask[y_w_min:y_w_max, best_x:min(width, best_x + 13)] > 0)

                if left_ok and right_ok:
                    if not any(abs(a["global_y"] - y_mid) < 25 for a in seam_anchors if a["type"] == "vertical" and abs(a["global_x"] - best_x) < 25):
                        seam_anchors.append(_create_anchor("vertical", best_x, y_mid, [t_left["tile_id"], t_right["tile_id"]]))

    # 2. Horizontal overlap bands (seams between adjacent rows r and r+1)
    for h_band in overlap_bands.get("horizontal_bands", []):
        r_top, r_bottom = h_band["row_pair"]
        y_seam = int(round((h_band["y1"] + h_band["y2"]) / 2.0))

        for c in range(grid_cols):
            t_top = tile_map.get((r_top, c))
            t_bottom = tile_map.get((r_bottom, c))
            if not t_top or not t_bottom:
                continue

            x_start = max(t_top["x1"], t_bottom["x1"])
            x_end = min(t_top["x2"], t_bottom["x2"])
            if x_end <= x_start + 10:
                continue

            y_a = max(0, y_seam - 4)
            y_b = min(height, y_seam + 5)
            strip = wire_mask[y_a:y_b, x_start:x_end]
            if strip.size == 0:
                continue

            col_profile = np.max(strip, axis=0)

            in_run = False
            start_x = 0
            for x_rel in range(len(col_profile)):
                if col_profile[x_rel] > 0 and not in_run:
                    in_run = True
                    start_x = x_rel
                elif col_profile[x_rel] == 0 and in_run:
                    in_run = False
                    x_mid = x_start + (start_x + x_rel - 1) // 2
                    local_x = (start_x + x_rel - 1) // 2
                    wire_y_offsets = np.where(strip[:, local_x] > 0)[0]
                    best_y = y_seam if len(wire_y_offsets) == 0 else y_a + int(np.mean(wire_y_offsets))

                    # Bilateral continuity check: wire must genuinely extend across both sides of the seam
                    x_w_min = max(0, x_mid - 6)
                    x_w_max = min(width, x_mid + 7)
                    top_ok = np.any(wire_mask[max(0, best_y - 12):best_y, x_w_min:x_w_max] > 0)
                    bottom_ok = np.any(wire_mask[best_y:min(height, best_y + 13), x_w_min:x_w_max] > 0)

                    if top_ok and bottom_ok:
                        if not any(abs(a["global_x"] - x_mid) < 25 for a in seam_anchors if a["type"] == "horizontal" and abs(a["global_y"] - best_y) < 25):
                            seam_anchors.append(_create_anchor("horizontal", x_mid, best_y, [t_top["tile_id"], t_bottom["tile_id"]]))

            if in_run:
                x_mid = x_start + (start_x + len(col_profile) - 1) // 2
                local_x = min(len(col_profile) - 1, (start_x + len(col_profile) - 1) // 2)
                wire_y_offsets = np.where(strip[:, local_x] > 0)[0]
                best_y = y_seam if len(wire_y_offsets) == 0 else y_a + int(np.mean(wire_y_offsets))

                x_w_min = max(0, x_mid - 6)
                x_w_max = min(width, x_mid + 7)
                top_ok = np.any(wire_mask[max(0, best_y - 12):best_y, x_w_min:x_w_max] > 0)
                bottom_ok = np.any(wire_mask[best_y:min(height, best_y + 13), x_w_min:x_w_max] > 0)

                if top_ok and bottom_ok:
                    if not any(abs(a["global_x"] - x_mid) < 25 for a in seam_anchors if a["type"] == "horizontal" and abs(a["global_y"] - best_y) < 25):
                        seam_anchors.append(_create_anchor("horizontal", x_mid, best_y, [t_top["tile_id"], t_bottom["tile_id"]]))

    return seam_anchors


def trace_wires_tiled(
    bgr_image: np.ndarray,
    tiles: List[Dict[str, Any]],
    overlap_bands: Dict[str, Any],
    detected_comps: List[Dict[str, Any]],
    pin_anchors: List[Dict[str, Any]],
    canny_low: int = 50,
    canny_high: int = 150,
    tracer_cfg: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[List[str]], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Executes wire tracing across overlapping image tiles using seam boundary anchors.
    Ensures that wires spanning across tile seams are bridged through physical anchor points
    located on the overlapping area boundaries.
    """
    from core.processing import TRANSPARENT_TYPES
    from wire_tracer.tracer import trace_wires
    from wire_tracer.config import WireTracerConfig

    if tracer_cfg is None:
        tracer_cfg = WireTracerConfig(canny_low=canny_low, canny_high=canny_high)

    height, width = bgr_image.shape[:2]

    # 1. Detect wire crossing points along overlap seams
    seam_anchors = detect_seam_wire_anchors(
        bgr_image=bgr_image,
        tiles=tiles,
        overlap_bands=overlap_bands,
        components=detected_comps,
        canny_low=canny_low,
        canny_high=canny_high,
    )
    print(f"[SEAM ANCHORS] Detected {len(seam_anchors)} seam wire anchor(s) across tile overlap boundaries.")

    # Group component pins by comp_idx
    comp_pins_map = defaultdict(list)
    for p in pin_anchors:
        if p.get("comp_idx", -1) >= 0:
            comp_pins_map[p["comp_idx"]].append(p)

    all_tile_nets: List[List[str]] = []
    tile_debug_info = {}

    # 2. Run wire tracing on each individual tile
    for t in tiles:
        tid = t["tile_id"]
        tx1, ty1, tx2, ty2 = t["x1"], t["y1"], t["x2"], t["y2"]
        tile_crop = bgr_image[ty1:ty2, tx1:tx2]
        if tile_crop.size == 0:
            continue

        tile_wt_components: List[Dict[str, Any]] = []

        # Add detected components whose bbox genuinely intersects this tile crop
        for idx, comp in enumerate(detected_comps):
            ctype = comp.get("type", "")
            if ctype in TRANSPARENT_TYPES:
                continue
            box = comp.get("box", [0, 0, 0, 0])
            bx, by, bw, bh = box[0], box[1], box[2], box[3]

            ix1 = max(bx, tx1)
            iy1 = max(by, ty1)
            ix2 = min(bx + bw, tx2)
            iy2 = min(by + bh, ty2)

            if ix2 > ix1 and iy2 > iy1:
                local_pins = []
                for p in comp_pins_map[idx]:
                    px, py = p["x"], p["y"]
                    if tx1 - 10 <= px <= tx2 + 10 and ty1 - 10 <= py <= ty2 + 10:
                        local_pins.append({
                            "id": f"{idx}_{p['pin_id']}",
                            "loc": [int(np.clip(px - tx1, 0, tx2 - tx1 - 1)), int(np.clip(py - ty1, 0, ty2 - ty1 - 1))]
                        })
                tile_wt_components.append({
                    "id": f"comp_{idx}",
                    "label": ctype,
                    "bbox": [int(ix1 - tx1), int(iy1 - ty1), int(ix2 - ix1), int(iy2 - iy1)],
                    "pins": local_pins
                })

        # Add seam anchors belonging to this tile
        for a in seam_anchors:
            if tid in a["tiles"]:
                loc = a["tile_coords"][tid]
                lx = int(np.clip(loc[0], 0, tx2 - tx1 - 1))
                ly = int(np.clip(loc[1], 0, ty2 - ty1 - 1))
                tile_wt_components.append({
                    "id": f"anc_{a['id']}",
                    "label": "junction",
                    "bbox": [max(0, lx - 2), max(0, ly - 2), 4, 4],
                    "pins": [{"id": a["id"], "loc": [lx, ly]}]
                })

        try:
            t_netlist, t_debug = trace_wires(tile_crop, tile_wt_components, config=tracer_cfg)
            all_tile_nets.extend(t_netlist)
            tile_debug_info[tid] = t_debug
        except Exception as e:
            print(f"[WARN] Wire tracing failed on tile {tid}: {e}")

    # 3. Global wire tracing pass (proven full-image tracing with real components)
    global_wt_components: List[Dict[str, Any]] = []
    for idx, comp in enumerate(detected_comps):
        ctype = comp.get("type", "")
        if ctype in TRANSPARENT_TYPES:
            continue
        pins = [{"id": f"{idx}_{p['pin_id']}", "loc": [p["x"], p["y"]]} for p in comp_pins_map[idx]]
        global_wt_components.append({
            "id": f"comp_{idx}",
            "label": ctype,
            "bbox": comp.get("box", [0, 0, 0, 0]),
            "pins": pins
        })

    try:
        global_netlist, global_debug = trace_wires(
            bgr_image, global_wt_components, config=tracer_cfg, method="connected_components", debug=True
        )
        all_tile_nets.extend(global_netlist)
    except Exception as ge:
        print(f"[WARN] Global wire tracing fallback: {ge}")
        global_debug = {"label_map": None, "pin_net_map": {}}

    if global_debug.get("label_map") is None and tile_debug_info:
        comp_map = np.zeros((height, width), dtype=np.int32)
        for t in tiles:
            tid = t["tile_id"]
            if tid in tile_debug_info and "label_map" in tile_debug_info[tid]:
                t_lm = tile_debug_info[tid]["label_map"]
                tx1, ty1, tx2, ty2 = t["x1"], t["y1"], t["x2"], t["y2"]
                mask = (t_lm > 0)
                sub = comp_map[ty1:ty2, tx1:tx2]
                sub[mask] = t_lm[mask]
        global_debug["label_map"] = comp_map

    # 4. Global Union-Find DSU across all tile and global nets
    parent: Dict[str, str] = {}
    def find(i: str) -> str:
        if i not in parent: parent[i] = i
        if parent[i] == i: return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i: str, j: str) -> None:
        ri, rj = find(i), find(j)
        if ri != rj: parent[ri] = rj

    for net in all_tile_nets:
        for k in range(len(net) - 1):
            union(net[k], net[k + 1])

    net_groups: Dict[str, List[str]] = defaultdict(list)
    for m in set(parent.keys()):
        net_groups[find(m)].append(m)

    # 5. Build clean, standard sequential connections between real component pins
    pin_lookup = {f"{p['comp_idx']}_{p['pin_id']}": p for p in pin_anchors if p.get("comp_idx", 0) >= 0}
    final_connections = []
    final_netlist = []

    for root, members in net_groups.items():
        # Keep ONLY real component pins (filter out seam bridge anchors)
        c_pins = [m for m in members if m in pin_lookup]

        if len(c_pins) < 2:
            continue

        c_pins = sorted(c_pins)
        final_netlist.append(c_pins)

        # Build sequential connections for the frontend A* router
        for i in range(len(c_pins) - 1):
            p1_idx, p1_pid = map(int, c_pins[i].split("_"))
            p2_idx, p2_pid = map(int, c_pins[i + 1].split("_"))
            final_connections.append({
                "pin1": {"comp_idx": p1_idx, "pin_id": p1_pid},
                "pin2": {"comp_idx": p2_idx, "pin_id": p2_pid}
            })

    final_netlist.sort(key=lambda pins: (-len(pins), pins[0] if pins else ""))

    # Update global_debug pin_net_map for frontend nodes assignment
    pin_net_map = {}
    for net_idx, net_pins in enumerate(final_netlist, start=1):
        for m in net_pins:
            pin_net_map[m] = net_idx
    global_debug["pin_net_map"] = pin_net_map

    return final_connections, pin_anchors, final_netlist, global_debug, seam_anchors


import time
import logging
import numpy as np
from wire_tracer.config import WireTracerConfig
from wire_tracer.preprocess import preprocess_image
from wire_tracer.masking import mask_components
from wire_tracer.nets import detect_nets_connected_components, build_netlist
# We will import skeleton functions later when implemented
from wire_tracer.pin_assign import assign_pins_to_nets
from wire_tracer.utils import validate_components, draw_debug_overlay

logger = logging.getLogger("wire_tracer")

def trace_wires(
    image_bgr: np.ndarray,
    components: list[dict],
    config: WireTracerConfig | None = None,
    method: str = "connected_components",
    debug: bool = False,
) -> tuple[list[list[str]], dict]:
    """
    Main entry point. Given a BGR circuit image and detected components,
    returns the netlist and a debug info dict.

    Args:
        image_bgr:   Raw BGR image
        components:  Detected component list
        config:      WireTracerConfig instance. If None, uses defaults.
        method:      "connected_components" (fast, recommended) or
                     "skeleton" (slower, distinguishes crossing types)
        debug:       If True, populates the returned debug dict with
                     intermediate images and stats.

    Returns:
        netlist:     list[list[str]] — the final connectivity netlist
        debug_info:  dict with keys:
                       "binary":         preprocessed binary image (if debug=True)
                       "wire_mask":      component-masked wire image (if debug=True)
                       "label_map":      colored label visualization (if debug=True)
                       "skeleton":       skeleton image (if debug=True, skeleton method only)
                       "pin_net_map":    dict mapping pin_id -> net_label
                       "num_nets":       int, number of distinct nets found
                       "floating_pins":  list of pin IDs not connected to any wire
                       "timing":         dict with per-stage wall-clock times in ms
    """
    # 1. Validate inputs
    errors = validate_components(components)
    if errors:
        raise ValueError("Component validation failed:\n" + "\n".join(errors))
        
    if not isinstance(image_bgr, np.ndarray):
        raise ValueError("image_bgr must be a numpy array")
        
    if config is None:
        config = WireTracerConfig()
        
    debug_info = {}
    timing = {}
    
    # 2. Preprocess
    t0 = time.time()
    binary = preprocess_image(
        image_bgr,
        blur_ksize=config.blur_ksize,
        adaptive_block_size=config.adaptive_block_size,
        adaptive_c=config.adaptive_c,
        morph_close_ksize=config.morph_close_ksize,
        morph_close_iterations=config.morph_close_iterations,
        min_blob_area=config.min_blob_area,
        canny_low=config.canny_low,
        canny_high=config.canny_high,
        canny_blur_ksize=config.canny_blur_ksize,
    )
    timing["preprocess"] = (time.time() - t0) * 1000
    if debug:
        debug_info["binary"] = binary.copy()
        
    # 3. Mask components
    t0 = time.time()
    wire_mask = mask_components(
        binary,
        components,
        pin_pad=config.pin_pad,
        erode_bbox=config.erode_bbox,
        enhance_wire_continuity=config.enhance_wire_continuity,
        h_wire_kernel_len=config.h_wire_kernel_len,
        v_wire_kernel_len=config.v_wire_kernel_len,
    )
    timing["mask_components"] = (time.time() - t0) * 1000
    if debug:
        debug_info["wire_mask"] = wire_mask.copy()
        
    # 4. Detect nets
    t0 = time.time()
    if method == "connected_components":
        num_nets, label_map = detect_nets_connected_components(
            wire_mask, 
            min_wire_area=config.min_wire_area
        )
    elif method == "skeleton":
        # Placeholder for skeleton logic
        from wire_tracer.skeleton import skeletonize_wires, classify_skeleton_pixels, build_skeleton_graph, nets_from_skeleton_graph
        skeleton = skeletonize_wires(wire_mask)
        if debug:
            debug_info["skeleton"] = skeleton.copy()
            
        class_res = classify_skeleton_pixels(skeleton)
        graph = build_skeleton_graph(skeleton, class_res, config.treat_crossings_as_junctions)
        num_nets, label_map = nets_from_skeleton_graph(graph, wire_mask.shape)
    else:
        raise ValueError(f"Unknown method '{method}'")
        
    timing["detect_nets"] = (time.time() - t0) * 1000
    if debug:
        debug_info["label_map"] = label_map.copy()
    debug_info["num_nets"] = num_nets
        
    # 5. Pin assignment
    t0 = time.time()
    pin_net_map = assign_pins_to_nets(
        label_map,
        components,
        max_search_radius=config.max_search_radius,
        strategy=config.pin_assignment_strategy
    )
    timing["pin_assign"] = (time.time() - t0) * 1000
    debug_info["pin_net_map"] = pin_net_map
    
    floating_pins = [pin_id for pin_id, net_id in pin_net_map.items() if net_id is None]
    debug_info["floating_pins"] = floating_pins
    
    # 6. Build Netlist
    t0 = time.time()
    netlist = build_netlist(
        pin_net_map,
        include_floating_pins=config.include_floating_pins
    )
    timing["build_netlist"] = (time.time() - t0) * 1000
    
    debug_info["timing"] = timing
    
    return netlist, debug_info

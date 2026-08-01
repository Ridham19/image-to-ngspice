from dataclasses import dataclass

@dataclass
class WireTracerConfig:
    # Preprocessing — dual-path binarization
    blur_ksize: int = 5
    adaptive_block_size: int = 15
    adaptive_c: int = 8
    morph_close_ksize: int = 7
    morph_close_iterations: int = 1
    min_blob_area: int = 20
    # Canny edge detection (Path B)
    canny_low: int = 50
    canny_high: int = 150
    canny_blur_ksize: int = 9

    # Masking
    pin_pad: int = 6
    erode_bbox: bool = True

    # Net detection
    min_wire_area: int = 25

    # Pin assignment
    max_search_radius: int = 80
    pin_assignment_strategy: str = "dominant"   # "dominant" or "nearest"

    # Wire continuity enhancement (post-masking)
    enhance_wire_continuity: bool = True
    h_wire_kernel_len: int = 40
    v_wire_kernel_len: int = 40

    # Skeleton-specific
    treat_crossings_as_junctions: bool = True

    # Output
    include_floating_pins: bool = False

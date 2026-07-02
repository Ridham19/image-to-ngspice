from dataclasses import dataclass

@dataclass
class WireTracerConfig:
    # Preprocessing
    blur_ksize: int = 3
    adaptive_block_size: int = 15
    adaptive_c: int = 4
    morph_close_ksize: int = 3
    morph_close_iterations: int = 1
    min_blob_area: int = 20

    # Masking
    pin_pad: int = 6
    erode_bbox: bool = True

    # Net detection
    min_wire_area: int = 50

    # Pin assignment
    max_search_radius: int = 20
    pin_assignment_strategy: str = "dominant"   # "dominant" or "nearest"

    # Skeleton-specific
    treat_crossings_as_junctions: bool = True

    # Output
    include_floating_pins: bool = False

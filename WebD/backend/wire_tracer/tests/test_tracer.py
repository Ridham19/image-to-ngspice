import pytest
import numpy as np
import cv2
from wire_tracer.tracer import trace_wires
from wire_tracer.config import WireTracerConfig
from wire_tracer.utils import draw_debug_overlay

def test_simple_series_circuit():
    # Draw a rectangle (closed loop) with 4 component bboxes at corners 
    # and 8 pins (2 per component at each corner). 
    # All pins should be in nets. The rectangle is one net, so each adjacent pair of pins should share a net.
    # We will make the image white, draw black lines.
    
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 0), 2)
    
    components = [
        # Top-left component
        {
            "id": "C1", "label": "R", "bbox": [40, 40, 20, 20],
            "pins": [
                {"id": "C1.p1", "loc": (60, 50)}, # Rightwards
                {"id": "C1.p2", "loc": (50, 60)}, # Downwards
            ]
        },
        # Top-right component
        {
            "id": "C2", "label": "R", "bbox": [140, 40, 20, 20],
            "pins": [
                {"id": "C2.p1", "loc": (140, 50)}, # Leftwards
                {"id": "C2.p2", "loc": (150, 60)}, # Downwards
            ]
        },
        # Bottom-right component
        {
            "id": "C3", "label": "R", "bbox": [140, 140, 20, 20],
            "pins": [
                {"id": "C3.p1", "loc": (150, 140)}, # Upwards
                {"id": "C3.p2", "loc": (140, 150)}, # Leftwards
            ]
        },
        # Bottom-left component
        {
            "id": "C4", "label": "R", "bbox": [40, 140, 20, 20],
            "pins": [
                {"id": "C4.p1", "loc": (60, 150)}, # Rightwards
                {"id": "C4.p2", "loc": (50, 140)}, # Upwards
            ]
        }
    ]
    
    # We expect 4 distinct nets: top edge, right edge, bottom edge, left edge.
    config = WireTracerConfig(pin_pad=2, min_blob_area=5)
    netlist, debug_info = trace_wires(img, components, config=config, debug=True)
    
    # Check that there are 4 nets
    assert debug_info["num_nets"] == 4
    # All pins connected
    assert len(debug_info["floating_pins"]) == 0
    assert len(netlist) == 4
    for net in netlist:
        assert len(net) == 2

def test_floating_pin_excluded_by_default():
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    # No wires
    components = [
        {
            "id": "C1", "label": "R", "bbox": [40, 40, 20, 20],
            "pins": [{"id": "C1.p1", "loc": (50, 50)}]
        }
    ]
    netlist, debug_info = trace_wires(img, components, debug=True)
    assert len(netlist) == 0
    assert len(debug_info["floating_pins"]) == 1

def test_floating_pin_included_when_configured():
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    components = [
        {
            "id": "C1", "label": "R", "bbox": [40, 40, 20, 20],
            "pins": [{"id": "C1.p1", "loc": (50, 50)}]
        }
    ]
    config = WireTracerConfig(include_floating_pins=True)
    netlist, debug_info = trace_wires(img, components, config=config, debug=True)
    assert len(netlist) == 1
    assert netlist[0] == ["C1.p1"]
    
def test_debug_overlay_returns_correct_shape():
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    # Draw a line
    cv2.line(img, (20, 50), (80, 50), (0, 0, 0), 2)
    components = [
        {
            "id": "C1", "label": "R", "bbox": [10, 40, 20, 20],
            "pins": [{"id": "C1.p1", "loc": (20, 50)}]
        }
    ]
    netlist, debug_info = trace_wires(img, components, debug=True)
    label_map = debug_info["label_map"]
    
    overlay = draw_debug_overlay(img, components, label_map, netlist)
    assert overlay.shape == img.shape
    assert overlay.dtype == img.dtype

def test_config_overrides_applied():
    # Pass a config with pin_pad=0 and verify masking behavior changes accordingly.
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    
    # We must draw a wire that crosses the bbox completely, 
    # so that the cleanup logic doesn't delete it as an isolated interior blob.
    # The bbox is 40..60. The wire goes from 30 to 70.
    cv2.line(img, (30, 50), (70, 50), (0, 0, 0), 2)
    components = [
        {
            "id": "C1", "label": "R", "bbox": [40, 40, 20, 20],
            "pins": [{"id": "C1.p1", "loc": (40, 50)}]
        }
    ]
    
    # If pin_pad=0, the entire bbox (x=40 to 60) is erased.
    # The remaining wire is x=30..39 and x=61..70. 
    # But wait! If we check inside the bbox region (40 to 60), it's empty!
    config_0 = WireTracerConfig(pin_pad=0, min_blob_area=1)
    _, debug_info_0 = trace_wires(img, components, config=config_0, debug=True)
    # The wire inside the bbox should be erased completely
    wire_in_bbox_0 = debug_info_0["wire_mask"][40:60, 40:60]
    assert np.count_nonzero(wire_in_bbox_0) == 0
    
    # If pin_pad=6, the erased region is only x=46 to 54.
    # So inside the bbox region (40 to 60), x=40..45 and x=55..60 will remain.
    config_6 = WireTracerConfig(pin_pad=6, min_blob_area=1)
    _, debug_info_6 = trace_wires(img, components, config=config_6, debug=True)
    wire_in_bbox_6 = debug_info_6["wire_mask"][40:60, 40:60]
    assert np.count_nonzero(wire_in_bbox_6) > 0

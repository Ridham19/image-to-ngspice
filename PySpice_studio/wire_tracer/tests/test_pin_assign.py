import pytest
import numpy as np
from wire_tracer.pin_assign import assign_pins_to_nets

def create_mock_components(pin_locs):
    pins = [{"id": f"C1.p{i}", "loc": loc} for i, loc in enumerate(pin_locs)]
    return [{"id": "C1", "label": "Mock", "bbox": [0, 0, 10, 10], "pins": pins}]

def test_pin_on_wire_assigns_correctly():
    # Place a pin exactly on a wire pixel. Expect correct label.
    label_map = np.zeros((50, 50), dtype=np.int32)
    label_map[25, 25] = 42
    
    components = create_mock_components([(25, 25)])
    
    # Test dominant
    result = assign_pins_to_nets(label_map, components, max_search_radius=5, strategy="dominant")
    assert result["C1.p0"] == 42
    
    # Test nearest
    result_nearest = assign_pins_to_nets(label_map, components, max_search_radius=5, strategy="nearest")
    assert result_nearest["C1.p0"] == 42

def test_pin_near_wire_assigns_within_radius():
    # Place a pin 5px away from a wire. Expect correct label with radius > 5.
    label_map = np.zeros((50, 50), dtype=np.int32)
    label_map[25, 30] = 7 # Wire is at x=30, y=25
    
    components = create_mock_components([(25, 25)]) # Pin is at x=25, y=25
    
    result = assign_pins_to_nets(label_map, components, max_search_radius=6, strategy="dominant")
    assert result["C1.p0"] == 7

def test_pin_too_far_returns_none():
    # Place a pin 30px from nearest wire with radius=20. Expect None.
    label_map = np.zeros((100, 100), dtype=np.int32)
    label_map[20, 20] = 1 # Wire at 20, 20
    
    components = create_mock_components([(50, 50)]) # Pin at 50, 50
    
    result = assign_pins_to_nets(label_map, components, max_search_radius=20, strategy="dominant")
    assert result["C1.p0"] is None

def test_two_pins_same_wire_same_net():
    # Two pins on the same wire blob. Expect same net label.
    label_map = np.zeros((50, 50), dtype=np.int32)
    label_map[10:20, 10] = 3 # vertical wire with label 3
    
    components = create_mock_components([(10, 12), (10, 18)])
    
    result = assign_pins_to_nets(label_map, components, max_search_radius=2)
    assert result["C1.p0"] == 3
    assert result["C1.p1"] == 3

def test_two_pins_different_wires_different_nets():
    # Two pins on unconnected wire blobs. Expect different labels.
    label_map = np.zeros((50, 50), dtype=np.int32)
    label_map[10, 10] = 1
    label_map[40, 40] = 2
    
    components = create_mock_components([(10, 10), (40, 40)])
    
    result = assign_pins_to_nets(label_map, components, max_search_radius=2)
    assert result["C1.p0"] == 1
    assert result["C1.p1"] == 2

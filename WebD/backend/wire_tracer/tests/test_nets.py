import pytest
import numpy as np
import cv2
from wire_tracer.nets import detect_nets_connected_components, build_netlist

def test_two_separate_wires_give_two_nets():
    # Draw two non-touching horizontal lines. Expect 2 nets.
    mask = np.zeros((50, 50), dtype=np.uint8)
    cv2.line(mask, (5, 10), (45, 10), 255, 2)
    cv2.line(mask, (5, 30), (45, 30), 255, 2)
    
    num_nets, label_map = detect_nets_connected_components(mask, min_wire_area=10)
    assert num_nets == 2
    assert label_map.max() == 2

def test_connected_l_shape_gives_one_net():
    # Draw an L-shaped wire. Expect 1 net.
    mask = np.zeros((50, 50), dtype=np.uint8)
    cv2.line(mask, (10, 10), (10, 40), 255, 2)
    cv2.line(mask, (10, 40), (40, 40), 255, 2)
    
    num_nets, label_map = detect_nets_connected_components(mask, min_wire_area=10)
    assert num_nets == 1

def test_noise_blobs_filtered_by_min_area():
    # Draw many tiny 1px blobs. With min_wire_area > 1, expect 0 nets.
    mask = np.zeros((50, 50), dtype=np.uint8)
    for i in range(10, 40, 5):
        mask[i, i] = 255
        
    num_nets, label_map = detect_nets_connected_components(mask, min_wire_area=5)
    assert num_nets == 0
    assert np.all(label_map == 0)

def test_t_junction_gives_one_net():
    # Draw three lines meeting at a T. Expect 1 net.
    mask = np.zeros((50, 50), dtype=np.uint8)
    cv2.line(mask, (10, 20), (40, 20), 255, 2)
    cv2.line(mask, (25, 20), (25, 40), 255, 2)
    
    num_nets, label_map = detect_nets_connected_components(mask, min_wire_area=10)
    assert num_nets == 1
    
def test_build_netlist():
    pin_net_map = {
        "R1.p1": 1,
        "C1.p1": 1,
        "R2.p1": 2,
        "C2.p1": 2,
        "U1.p1": 2,
        "R3.p1": None
    }
    
    # Exclude floating
    netlist = build_netlist(pin_net_map, include_floating_pins=False)
    assert len(netlist) == 2
    assert ["C2.p1", "R2.p1", "U1.p1"] in netlist # size 3
    assert ["C1.p1", "R1.p1"] in netlist # size 2
    
    # Include floating
    netlist_float = build_netlist(pin_net_map, include_floating_pins=True)
    assert len(netlist_float) == 3
    assert ["R3.p1"] in netlist_float

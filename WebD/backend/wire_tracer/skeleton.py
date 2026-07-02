import numpy as np
import cv2
import networkx as nx
from skimage.morphology import skeletonize

def skeletonize_wires(wire_mask: np.ndarray) -> np.ndarray:
    """
    Thin all wire blobs to 1-pixel-wide skeleton using Zhang-Suen thinning.
    Uses skimage.morphology.skeletonize.

    Returns:
        skeleton: np.ndarray, shape (H, W), dtype uint8, values {0, 255}
    """
    if not isinstance(wire_mask, np.ndarray):
        raise ValueError("wire_mask must be a numpy array")
        
    bool_mask = wire_mask > 0
    skel = skeletonize(bool_mask)
    return (skel * 255).astype(np.uint8)

def classify_skeleton_pixels(skeleton: np.ndarray) -> dict[str, np.ndarray]:
    """
    For every foreground pixel in the skeleton, count its 8-connected neighbors
    and classify it.

    Returns dict with keys 'endpoints', 't_junctions', 'crossings', each a boolean mask.
    """
    if not isinstance(skeleton, np.ndarray):
        raise ValueError("skeleton must be a numpy array")
        
    skel_binary = (skeleton > 0).astype(np.int32)
    
    # Use cv2.filter2D to count 8-connected neighbors
    kernel = np.ones((3, 3), dtype=np.int32)
    # filter2D includes the center pixel, so we subtract skel_binary
    neighbor_count = cv2.filter2D(skel_binary, -1, kernel) - skel_binary
    
    # We only care about foreground pixels
    endpoints = (skel_binary == 1) & (neighbor_count == 1)
    t_junctions = (skel_binary == 1) & (neighbor_count == 3)
    crossings = (skel_binary == 1) & (neighbor_count >= 4)
    
    return {
        'endpoints': endpoints,
        't_junctions': t_junctions,
        'crossings': crossings
    }

def build_skeleton_graph(
    skeleton: np.ndarray,
    classify_result: dict[str, np.ndarray],
    treat_crossings_as_junctions: bool = True,
) -> nx.Graph:
    """
    Build a graph where:
        Nodes = endpoint pixels + junction pixels
        Edges = skeleton branches connecting them
    """
    if not isinstance(skeleton, np.ndarray):
        raise ValueError("skeleton must be a numpy array")
        
    skel_binary = (skeleton > 0).astype(np.uint8)
    h, w = skel_binary.shape
    
    endpoints = classify_result['endpoints']
    t_junctions = classify_result['t_junctions']
    crossings = classify_result['crossings']
    
    if treat_crossings_as_junctions:
        junctions = t_junctions | crossings
    else:
        junctions = t_junctions
        
    nodes = endpoints | junctions
    
    G = nx.Graph()
    
    # Add nodes to graph
    node_coords = np.argwhere(nodes)
    for y, x in node_coords:
        if endpoints[y, x]:
            ntype = 'endpoint'
        elif t_junctions[y, x]:
            ntype = 't_junction'
        else:
            ntype = 'crossing'
        G.add_node((y, x), pixel=(y, x), type=ntype)
        
    # Helper to get valid 8-neighbors
    def get_neighbors(y, x):
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and skel_binary[ny, nx]:
                    neighbors.append((ny, nx))
        return neighbors

    visited = set()
    
    # Trace edges from each node
    for start_node in G.nodes():
        sy, sx = start_node
        
        for ny, nx in get_neighbors(sy, sx):
            # A branch is defined by a start node and an initial direction
            # If we haven't traced this exact undirected segment yet
            edge_id1 = (start_node, (ny, nx))
            edge_id2 = ((ny, nx), start_node)
            if edge_id1 in visited or edge_id2 in visited:
                continue
                
            visited.add(edge_id1)
            visited.add(edge_id2)
            
            # Trace until we hit another node or a dead end
            path = [(sy, sx)]
            curr_y, curr_x = ny, nx
            prev_y, prev_x = sy, sx
            
            while True:
                path.append((curr_y, curr_x))
                if (curr_y, curr_x) in G.nodes():
                    # Reached another node, finish edge
                    G.add_edge(start_node, (curr_y, curr_x), path=path, length=len(path))
                    break
                    
                # Continue tracing
                next_pixels = []
                for nny, nnx in get_neighbors(curr_y, curr_x):
                    if (nny, nnx) != (prev_y, prev_x):
                        next_pixels.append((nny, nnx))
                        
                if not next_pixels:
                    # Dead end, shouldn't happen unless isolated or weird artifact, just ignore
                    break
                    
                # Mark transition as visited
                edge_id1 = ((curr_y, curr_x), next_pixels[0])
                edge_id2 = (next_pixels[0], (curr_y, curr_x))
                visited.add(edge_id1)
                visited.add(edge_id2)
                
                prev_y, prev_x = curr_y, curr_x
                curr_y, curr_x = next_pixels[0]

    return G

def nets_from_skeleton_graph(graph: nx.Graph) -> tuple[int, np.ndarray]:
    """
    Find connected components in the skeleton graph and produce a label_map.
    """
    # Initialize empty label_map. We need the shape, but graph doesn't store it directly.
    # However, node attributes have 'pixel', we can't infer max bounds easily unless we find the max.
    # But wait, label_map is usually the size of the image. 
    # For now, let's find max x and max y from nodes and paths to create a bounding label_map.
    # A better way is to pass the original shape, but the signature is strict.
    
    max_y, max_x = 0, 0
    for u, v, data in graph.edges(data=True):
        for y, x in data.get('path', []):
            if y > max_y: max_y = y
            if x > max_x: max_x = x
            
    for y, x in graph.nodes():
        if y > max_y: max_y = y
        if x > max_x: max_x = x
        
    label_map = np.zeros((max_y + 1, max_x + 1), dtype=np.int32)
    
    connected_components = list(nx.connected_components(graph))
    num_nets = len(connected_components)
    
    for i, comp in enumerate(connected_components):
        label = i + 1
        # Draw the nodes
        for node in comp:
            label_map[node[0], node[1]] = label
            
        # Draw the edges
        subgraph = graph.subgraph(comp)
        for u, v, data in subgraph.edges(data=True):
            for y, x in data.get('path', []):
                label_map[y, x] = label
                
    return num_nets, label_map

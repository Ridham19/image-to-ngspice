import numpy as np
from library import DB

def analyze_circuit(components, wires):
    adj = {}
    for p1, p2 in wires:
        p1 = (round(p1[0]), round(p1[1])); p2 = (round(p2[0]), round(p2[1]))
        if p1 not in adj: adj[p1] = []
        if p2 not in adj: adj[p2] = []
        adj[p1].append(p2); adj[p2].append(p1)

    for c in components:
        for p in c.get_pins():
            pt = (round(p[0]), round(p[1]))
            if pt not in adj: adj[pt] = [] 

    node_map = {} 
    visited = set()
    node_counter = 1
    gnd_locs = set(); label_locs = {} 

    for c in components:
        pt = c.get_pins()[0]; coord = (round(pt[0]), round(pt[1]))
        if c.type == 'gnd': gnd_locs.add(coord)
        if c.type == 'label': label_locs[coord] = c.value 

    for pt in list(adj.keys()):
        if pt not in visited:
            cid = str(node_counter); node_counter += 1
            stack = [pt]; cluster = []; is_gnd = False; custom_name = None
            while stack:
                curr = stack.pop()
                if curr in visited: continue
                visited.add(curr); cluster.append(curr)
                if curr in gnd_locs: is_gnd = True
                if curr in label_locs: custom_name = label_locs[curr]
                if curr in adj:
                    for n in adj[curr]: stack.append(n)
            
            final_id = '0' if is_gnd else (custom_name if custom_name else cid)
            for p in cluster: node_map[p] = final_id
            
    unique_nodes = sorted(list(set(node_map.values())))
    
    # Updated Lists to include 'sine_source'
    all_source_types = ['source', 'current', 'ac_source', 'sine_source', 'pulse']
    
    sources = [c.name for c in components if c.type in all_source_types]
    sweepables = [c.name for c in components if c.type in all_source_types + ['resistor']]
    
    return node_map, sources, unique_nodes, sweepables

def generate_netlist(components, wires, sim_data):
    node_map, _, _, _ = analyze_circuit(components, wires)
    lines = ["* PySpice Studio Netlist", ""]

    models_needed = set()
    for c in components:
        if c.type == 'diode': models_needed.add(".Model Dx diode (Is=14n Rs=0 N=1)")
        if c.type == 'bjt_npn': models_needed.add(".Model Tx NPN (BF=300)")
        if c.type == 'bjt_pnp': models_needed.add(".Model Tx_pnp PNP (BF=300)")
        # Removed Mosfet model since we removed component
    
    for m in models_needed: lines.append(m)
    lines.append("")

    for c in components:
        if c.type in ['gnd', 'label']: continue
        pins = c.get_pins()
        nodes = []
        for p in pins:
            pt = (round(p[0]), round(p[1]))
            nodes.append(node_map.get(pt, f"NC"))

        if c.type in DB:
            fmt = DB[c.type]['spice']
            ctx = {'name': c.name, 'n1': nodes[0] if len(nodes)>0 else '0', 'n2': nodes[1] if len(nodes)>1 else '0', 'n3': nodes[2] if len(nodes)>2 else '0', 'n4': nodes[3] if len(nodes)>3 else '0'}
            for k, v in c.params.items(): ctx[k] = v
            try: lines.append(fmt.format(**ctx))
            except: pass

    lines.append("")
    lines.append(sim_data.get('cmd', '.op'))
    
    lines.append(".control")
    lines.append("run")
    
    colors = sim_data.get('colors', {})
    if not colors: 
        lines.append("set color0 = white")
        lines.append("set color1 = black") 
    else:
        for idx, col in colors.items(): lines.append(f"set color{idx} = {col}")
    
    lines.append("set xbrushwidth = 2")
    
    plots = sim_data.get('plots', {})
    if plots:
        sorted_wins = sorted(plots.keys())
        for win_id in sorted_wins:
            sigs = " ".join(plots[win_id])
            lines.append(f"plot {sigs} title 'Graph Window {win_id}'")
    elif 'plot' in sim_data and sim_data['plot']:
        lines.append(f"plot {sim_data['plot']}")
    
    lines.append(".endc")
    lines.append(".end")
    return "\n".join(lines)
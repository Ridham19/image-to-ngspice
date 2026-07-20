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
    
    # Updated Lists to include all source types
    all_source_types = [
        'source', 'current', 'ac_source', 'sine_source', 'pulse',
        'vss', 'current_source', 'pulse_source', 'sine_source_td',
        'exp_source', 'pwl_source', 'sffm_source', 'am_source'
    ]
    
    sources = [c.name for c in components if c.type in all_source_types]
    sweepables = [c.name for c in components if c.type in all_source_types + ['resistor']]
    
    return node_map, sources, unique_nodes, sweepables

def generate_netlist(components, wires, sim_data):
    node_map, _, _, _ = analyze_circuit(components, wires)
    lines = ["* PySpice Studio Netlist", ""]

    models_needed = set()
    subckts_needed = {}  # key -> definition text
    
    for c in components:
        if c.type in ['diode', 'diode_led', 'diode_zener']:
            if c.type == 'diode_zener':
                models_needed.add(".Model Dx diode (Is=14n Rs=0 N=1 BV=5.1)")
            else:
                models_needed.add(".Model Dx diode (Is=14n Rs=0 N=1)")
        elif c.type in ['bjt_npn', 'phototransistor']:
            models_needed.add(".Model Tx NPN (BF=300)")
        elif c.type == 'bjt_pnp':
            models_needed.add(".Model Tx_pnp PNP (BF=300)")
        elif c.type in ['mosfet', 'nmos']:
            model_param = c.params.get('model', 'Mx' if c.type == 'mosfet' else 'nmos')
            models_needed.add(f".model {model_param} nmos level=54")
        elif c.type == 'pmos':
            model_param = c.params.get('model', 'pmos')
            models_needed.add(f".model {model_param} pmos level=54")
        elif c.type == 'opamp':
            model_name = c.params.get('model', 'LM741')
            if model_name == 'LM741' and 'opamp' not in subckts_needed:
                subckts_needed['opamp'] = """* --- Ideal Op-Amp Subcircuit: LM741 ---
.subckt LM741 vplus vminus out vspos vsneg
Rin vplus vminus 2MEG
Eout int 0 vplus vminus 200000
Rout_int int out 75
.ends LM741"""
        elif c.type == 'ic':
            custom_body = c.params.get('custom_subckt', '').strip()
            subckt_name = c.params.get('subckt_name', 'MyIC')
            key = f'ic_{subckt_name}'
            if custom_body and key not in subckts_needed:
                subckts_needed[key] = f"\n* --- User-defined subcircuit: {subckt_name} ---\n{custom_body}"
    
    for defn in subckts_needed.values():
        lines.append(defn)
    if subckts_needed:
        lines.append("")

    for m in sorted(models_needed):
        lines.append(m)
    if models_needed:
        lines.append("")

    SPICE_PREFIX_ORDER = {'C': 0, 'D': 1, 'I': 2, 'K': 3, 'L': 4, 'Q': 5, 'R': 6, 'V': 7, 'X': 8}
    device_lines = []

    for c in components:
        if c.type in ['gnd', 'label']: continue
        pins = c.get_pins()
        nodes = []
        for p in pins:
            pt = (round(p[0]), round(p[1]))
            nodes.append(node_map.get(pt, f"NC"))

        # Op-Amp Translation
        if c.type == 'opamp':
            n_vplus  = nodes[0] if len(nodes) > 0 else 'NC'
            n_vminus = nodes[1] if len(nodes) > 1 else 'NC'
            n_out    = nodes[2] if len(nodes) > 2 else 'NC'
            vs_pos   = c.params.get('vs_pos', '15')
            vs_neg   = c.params.get('vs_neg', '-15')
            model    = c.params.get('model', 'LM741')
            vs_pos_node = f"vsp_{c.name}"
            vs_neg_node = f"vsn_{c.name}"
            device_lines.append((SPICE_PREFIX_ORDER.get('V', 7), f"VsPos_{c.name}",
                f"VsPos_{c.name} {vs_pos_node} 0 DC {vs_pos}"))
            device_lines.append((SPICE_PREFIX_ORDER.get('V', 7), f"VsNeg_{c.name}",
                f"VsNeg_{c.name} {vs_neg_node} 0 DC {vs_neg}"))
            line = f"X{c.name} {n_vplus} {n_vminus} {n_out} {vs_pos_node} {vs_neg_node} {model}"
            device_lines.append((SPICE_PREFIX_ORDER.get('X', 8), f"X{c.name}", line))
            continue

        # IC Translation
        if c.type == 'ic':
            subckt_name = c.params.get('subckt_name', 'MyIC')
            node_list = " ".join(nodes) if nodes else "NC"
            line = f"X{c.name} {node_list} {subckt_name}"
            device_lines.append((SPICE_PREFIX_ORDER.get('X', 8), f"X{c.name}", line))
            continue

        # Transformer Translation
        if c.type == 'transformer':
            n1 = nodes[0] if len(nodes) > 0 else 'NC'
            n2 = nodes[1] if len(nodes) > 1 else 'NC'
            n3 = nodes[2] if len(nodes) > 2 else 'NC'
            n4 = nodes[3] if len(nodes) > 3 else 'NC'
            inductance = c.params.get('value', '1m')
            coupling   = c.params.get('coupling', '0.99')
            la_name = f"L{c.name}a"
            lb_name = f"L{c.name}b"
            k_name  = f"K{c.name}"
            device_lines.append((SPICE_PREFIX_ORDER.get('L', 4), la_name, f"{la_name} {n1} {n2} {inductance}"))
            device_lines.append((SPICE_PREFIX_ORDER.get('L', 4), lb_name, f"{lb_name} {n3} {n4} {inductance}"))
            device_lines.append((SPICE_PREFIX_ORDER.get('K', 3), k_name, f"{k_name} {la_name} {lb_name} {coupling}"))
            continue

        if c.type in DB:
            fmt = DB[c.type]['spice']
            ctx = {
                'name': c.name,
                'n1': nodes[0] if len(nodes)>0 else '0',
                'n2': nodes[1] if len(nodes)>1 else '0',
                'n3': nodes[2] if len(nodes)>2 else '0',
                'n4': nodes[3] if len(nodes)>3 else '0',
                'w': '10u',
                'l': '0.18u'
            }
            for k, v in c.params.items(): ctx[k] = v
            try:
                line = fmt.format(**ctx)
                prefix = c.name[0].upper() if c.name else 'Z'
                sort_key = SPICE_PREFIX_ORDER.get(prefix, 99)
                device_lines.append((sort_key, c.name, line))
            except Exception as e:
                device_lines.append((99, c.name, f"* ERROR: Missing param {e} for {c.name}"))

    device_lines.sort(key=lambda t: (t[0], t[1]))
    for _, _, line in device_lines:
        lines.append(line)

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
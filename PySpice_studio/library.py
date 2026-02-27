import schemdraw.elements as elm

# Safe Import
def get_mosfet(type_='n'):
    opts = ['Mosfet', 'MosfetNg', 'NFet', 'FET']
    for o in opts: 
        if hasattr(elm, o): return getattr(elm, o)
    return elm.Resistor

def get_jfet(type_='n'):
    opts = ['JFet', 'JFetN', 'JFetNg']
    for o in opts: 
        if hasattr(elm, o): return getattr(elm, o)
    return elm.Resistor

# --- COMPONENT DATABASE ---
DB = {
    # PASSIVES
    'resistor': {
        'label': 'Resistor', 'prefix': 'R', 'category': 'Passives',
        'element': elm.Resistor, 'shape': '2_pin',
        'params': {'value': '1k'}, 'spice': "{name} {n1} {n2} {value}",
        'btn_text': 'Res'
    },
    'capacitor': {
        'label': 'Capacitor', 'prefix': 'C', 'category': 'Passives',
        'element': elm.Capacitor, 'shape': '2_pin_short',
        'params': {'value': '1u', 'ic': '0'}, 'spice': "{name} {n1} {n2} {value} ic={ic}",
        'btn_text': 'Cap'
    },
    'inductor': {
        'label': 'Inductor', 'prefix': 'L', 'category': 'Passives',
        'element': elm.Inductor2, 'shape': '2_pin',
        'params': {'value': '1m', 'ic': '0'}, 'spice': "{name} {n1} {n2} {value} ic={ic}",
        'btn_text': 'Ind'
    },
    'diode': {
        'label': 'Diode', 'prefix': 'D', 'category': 'Passives',
        'element': elm.Diode, 'shape': '2_pin',
        'params': {'model': 'Dx'}, 'spice': "{name} {n1} {n2} {model}",
        'btn_text': 'Diode'
    },

    # SOURCES
    'source': {
        'label': 'DC Voltage', 'prefix': 'V', 'category': 'Sources',
        'element': elm.SourceV, 'shape': 'v_source',
        'params': {'dc': '5'}, 'spice': "{name} {n1} {n2} DC {dc}",
        'btn_text': 'Vdc'
    },
    'current': {
        'label': 'DC Current', 'prefix': 'I', 'category': 'Sources',
        'element': elm.SourceI, 'shape': 'v_source',
        'params': {'dc': '1m'}, 'spice': "{name} {n1} {n2} DC {dc}",
        'btn_text': 'Idc'
    },
    
    # --- UPDATED AC SOURCE (Frequency Domain) ---
    'ac_source': {
        'label': 'AC Source', 'prefix': 'V', 'category': 'Sources',
        'element': elm.SourceSin, 'shape': 'v_source',
        'params': {'mag': '1', 'phase': '0'}, 
        'spice': "{name} {n1} {n2} AC {mag} {phase}",
        'btn_text': 'Vac'
    },
    
    # --- NEW SINE SOURCE (Time Domain) ---
    'sine_source': {
        'label': 'Sine Wave', 'prefix': 'V', 'category': 'Sources',
        'element': elm.SourceSin, 'shape': 'v_source',
        'params': {'offset': '0', 'amp': '1', 'freq': '1k'}, 
        'spice': "{name} {n1} {n2} SIN({offset} {amp} {freq})",
        'btn_text': 'Sine'
    },

    'pulse': {
        'label': 'Pulse', 'prefix': 'V', 'category': 'Sources',
        'element': elm.SourcePulse, 'shape': 'v_source',
        'params': {'v1': '0', 'v2': '5', 'td': '10m', 'tr': '1n', 'tf': '1n', 'pw': '10m', 'per': '20m'},
        'spice': "{name} {n1} {n2} PULSE({v1} {v2} {td} {tr} {tf} {pw} {per})",
        'btn_text': 'Pulse'
    },
    'gnd': {
        'label': 'Ground', 'prefix': '0', 'category': 'Sources',
        'element': elm.Ground, 'shape': '1_pin',
        'params': {}, 'spice': "",
        'btn_text': 'GND'
    },

    'label': {
        'label': 'Node Label', 'prefix': 'Net', 'category': 'Other',
        'element': elm.Dot, 'shape': 'label', 
        'params': {'name': 'OUT'}, 'spice': "",
        'btn_text': 'Label'
    },

    # ACTIVE
    'bjt_npn': {
        'label': 'NPN BJT', 'prefix': 'Q', 'category': 'Active',
        'element': elm.BjtNpn, 'shape': '3_pin_bjt',
        'params': {'model': 'Tx'}, 'spice': "{name} {n2} {n1} {n3} {model}",
        'btn_text': 'NPN'
    },
    'bjt_pnp': {
        'label': 'PNP BJT', 'prefix': 'Q', 'category': 'Active',
        'element': elm.BjtPnp, 'shape': '3_pin_bjt',
        'params': {'model': 'Tx_pnp'}, 'spice': "{name} {n2} {n1} {n3} {model}",
        'btn_text': 'PNP'
    }
}
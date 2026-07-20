import schemdraw.elements as elm

# Safe Import
def get_mosfet(type_='n'):
    opts = ['NFet', 'PFet', 'Mosfet', 'MosfetNg', 'FET'] if type_ == 'n' else ['PFet', 'NFet', 'Mosfet', 'PFetNg', 'FET']
    # Let's adjust opts to make sure nmos gets nfet/mosfet and pmos gets pfet/pmosfet
    if type_ == 'p':
        opts = ['PFet', 'Mosfet', 'PFetNg', 'FET']
    else:
        opts = ['NFet', 'Mosfet', 'NFetNg', 'FET']
    for o in opts: 
        if hasattr(elm, o): return getattr(elm, o)
    return elm.Resistor

def get_jfet(type_='n'):
    opts = ['JFet', 'JFetN', 'JFetNg']
    for o in opts: 
        if hasattr(elm, o): return getattr(elm, o)
    return elm.Resistor

def get_polarized_cap():
    for o in ['CapacitorElectrolytic', 'CapElectrolytic', 'Capacitor']:
        if hasattr(elm, o): return getattr(elm, o)
    return elm.Capacitor

def get_photoresistor():
    for o in ['ResistorLdr', 'LDR', 'Resistor']:
        if hasattr(elm, o): return getattr(elm, o)
    return elm.Resistor

def get_led():
    for o in ['LED', 'DiodeLed', 'Diode']:
        if hasattr(elm, o): return getattr(elm, o)
    return elm.Diode

def get_zener():
    for o in ['DiodeZener', 'Zener', 'Diode']:
        if hasattr(elm, o): return getattr(elm, o)
    return elm.Diode

def get_phototransistor():
    # Phototransistor often uses phototransistor or standard BjtNpn
    for o in ['BjtNpn', 'Resistor']:
        if hasattr(elm, o): return getattr(elm, o)
    return elm.Resistor

def get_opamp():
    for o in ['Opamp', 'OpAmp']:
        if hasattr(elm, o): return getattr(elm, o)
    return elm.Resistor

def get_transformer():
    for o in ['Transformer', 'Transformer4p', 'Resistor']:
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
    'junction': {
        'label': 'Junction', 'prefix': 'J', 'category': 'Other',
        'element': elm.Dot, 'shape': 'label',
        'params': {}, 'spice': "",
        'btn_text': 'Junc'
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
    },
    'capacitor_polarized': {
        'label': 'Polarized Cap', 'prefix': 'C', 'category': 'Passives',
        'element': get_polarized_cap(), 'shape': '2_pin_short',
        'params': {'value': '10u', 'ic': '0'}, 'spice': "{name} {n1} {n2} {value} ic={ic}",
        'btn_text': 'PolCap'
    },
    'resistor_photo': {
        'label': 'Photoresistor', 'prefix': 'R', 'category': 'Passives',
        'element': get_photoresistor(), 'shape': '2_pin',
        'params': {'value': '10k'}, 'spice': "{name} {n1} {n2} {value}",
        'btn_text': 'LDR'
    },
    'diode_led': {
        'label': 'LED', 'prefix': 'D', 'category': 'Passives',
        'element': get_led(), 'shape': '2_pin',
        'params': {'model': 'Dx'}, 'spice': "{name} {n1} {n2} {model}",
        'btn_text': 'LED'
    },
    'diode_zener': {
        'label': 'Zener Diode', 'prefix': 'D', 'category': 'Passives',
        'element': get_zener(), 'shape': '2_pin',
        'params': {'model': 'Dx'}, 'spice': "{name} {n1} {n2} {model}",
        'btn_text': 'Zener'
    },
    'phototransistor': {
        'label': 'Phototransistor', 'prefix': 'Q', 'category': 'Active',
        'element': get_phototransistor(), 'shape': '3_pin_bjt',
        'params': {'model': 'Tx'}, 'spice': "{name} {n2} {n1} {n3} {model}",
        'btn_text': 'PhotoQ'
    },
    'opamp': {
        'label': 'Op-Amp', 'prefix': 'X', 'category': 'Active',
        'element': get_opamp(), 'shape': 'opamp',
        'params': {'model': 'LM741', 'vs_pos': '15', 'vs_neg': '-15'}, 'spice': "",
        'btn_text': 'OpAmp'
    },
    'ic': {
        'label': 'IC', 'prefix': 'X', 'category': 'Active',
        'element': elm.Resistor, 'shape': 'ic',
        'params': {'subckt_name': 'MyIC', 'num_pins': '2', 'custom_subckt': ''}, 'spice': "",
        'btn_text': 'IC'
    },
    'transformer': {
        'label': 'Transformer', 'prefix': 'T', 'category': 'Passives',
        'element': get_transformer(), 'shape': 'transformer',
        'params': {'value': '1m', 'coupling': '0.99'}, 'spice': "",
        'btn_text': 'Xform'
    },
    'nmos': {
        'label': 'N-MOSFET', 'prefix': 'M', 'category': 'Active',
        'element': get_mosfet('n'), 'shape': '3_pin_fet',
        'params': {'model': 'nmos', 'w': '10u', 'l': '0.18u'}, 'spice': "{name} {n2} {n1} {n3} {n3} {model} w={w} l={l}",
        'btn_text': 'NMOS'
    },
    'pmos': {
        'label': 'P-MOSFET', 'prefix': 'M', 'category': 'Active',
        'element': get_mosfet('p'), 'shape': '3_pin_fet',
        'params': {'model': 'pmos', 'w': '10u', 'l': '0.18u'}, 'spice': "{name} {n2} {n1} {n3} {n3} {model} w={w} l={l}",
        'btn_text': 'PMOS'
    },
    'vss': {
        'label': 'VSS Supply', 'prefix': 'V', 'category': 'Sources',
        'element': elm.SourceV, 'shape': 'v_source',
        'params': {'dc': '-5'}, 'spice': "{name} {n1} {n2} DC {dc}",
        'btn_text': 'Vss'
    },
    'current_source': {
        'label': 'DC Current', 'prefix': 'I', 'category': 'Sources',
        'element': elm.SourceI, 'shape': 'v_source',
        'params': {'dc': '1m'}, 'spice': "{name} {n1} {n2} DC {dc}",
        'btn_text': 'Idc'
    },
    'pulse_source': {
        'label': 'Pulse Source', 'prefix': 'V', 'category': 'Sources',
        'element': elm.SourcePulse, 'shape': 'v_source',
        'params': {'v1': '0', 'v2': '5', 'td': '0', 'tr': '1n', 'tf': '1n', 'pw': '10u', 'per': '20u'},
        'spice': "{name} {n1} {n2} PULSE({v1} {v2} {td} {tr} {tf} {pw} {per})",
        'btn_text': 'PulseSrc'
    },
    'sine_source_td': {
        'label': 'Sine Source', 'prefix': 'V', 'category': 'Sources',
        'element': elm.SourceSin, 'shape': 'v_source',
        'params': {'vo': '0', 'va': '5', 'freq': '1k', 'td': '0', 'theta': '0', 'phase': '0'},
        'spice': "{name} {n1} {n2} SINE({vo} {va} {freq} {td} {theta} {phase})",
        'btn_text': 'SineSrc'
    },
    'exp_source': {
        'label': 'Exp Source', 'prefix': 'V', 'category': 'Sources',
        'element': elm.SourceV, 'shape': 'v_source',
        'params': {'v1': '0', 'v2': '5', 'td1': '2u', 'tau1': '2u', 'td2': '5u', 'tau2': '5u'},
        'spice': "{name} {n1} {n2} EXP({v1} {v2} {td1} {tau1} {td2} {tau2})",
        'btn_text': 'Exp'
    },
    'pwl_source': {
        'label': 'PWL Source', 'prefix': 'V', 'category': 'Sources',
        'element': elm.SourceV, 'shape': 'v_source',
        'params': {'pwl_data': '0 0 1m 5'},
        'spice': "{name} {n1} {n2} PWL({pwl_data})",
        'btn_text': 'PWL'
    },
    'sffm_source': {
        'label': 'SFFM Source', 'prefix': 'V', 'category': 'Sources',
        'element': elm.SourceV, 'shape': 'v_source',
        'params': {'vo': '0', 'va': '1', 'fc': '1k', 'mdi': '5', 'fs': '200'},
        'spice': "{name} {n1} {n2} SFFM({vo} {va} {fc} {mdi} {fs})",
        'btn_text': 'FM'
    },
    'am_source': {
        'label': 'AM Source', 'prefix': 'V', 'category': 'Sources',
        'element': elm.SourceV, 'shape': 'v_source',
        'params': {'va': '5', 'fc': '1k', 'mf': '100', 'ph': '0'},
        'spice': "{name} {n1} {n2} AM({va} {fc} {mf} {ph})",
        'btn_text': 'AM'
    }
}
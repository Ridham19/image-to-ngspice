"""
OCR Value Validator & Autocorrect Engine for PySpice Studio
File: PySpice_studio/core/ocr_validator.py

This module provides validation, sanitization, handwriting digit error correction,
unit normalization, and default fallback estimation for hand-drawn electronic component OCR values.

Users can directly modify rules, regexes, unit mappings, and standard defaults in this file.
"""
import re
from typing import Tuple, Dict, Any, Optional

# ═══════════════════════════════════════════════════════════════════════
# DEFAULT FALLBACK VALUES
# ═══════════════════════════════════════════════════════════════════════
DEFAULT_COMPONENT_VALUES: Dict[str, str] = {
    'resistor':             '1k',
    'resistor_photo':       '10k',
    'capacitor':            '1u',
    'capacitor_polarized':  '10u',
    'inductor':             '1m',
    'source':               '5',
    'voltage_source':       '5',
    'ac_source':            '1',
    'current_source':       '1m',
    'vss':                  '-5',
    'pulse_source':         'PULSE(0 5 0 1n 1n 10u 20u)',
    'sine_source':          'SINE(0 1 1k)',
    'exp_source':           'EXP(0 5 1u 10u 20u 50u)',
    'pwl_source':           '0 0 10u 5',
    'sffm_source':          '0 1 1k 5 200',
    'am_source':            '5 1k 100 0',
    'diode':                'Dx',
    'diode_led':            'Dx',
    'diode_zener':          'Dx',
    'bjt':                  'Tx',
    'bjt_npn':              'Tx',
    'bjt_pnp':              'Tx_pnp',
    'mosfet':               'nmos',
    'nmos':                 'nmos',
    'pmos':                 'pmos',
}

# ═══════════════════════════════════════════════════════════════════════
# HANDWRITING DIGIT SWAPS & SUFFIX CORRECTION MAPS
# ═══════════════════════════════════════════════════════════════════════
HANDWRITING_DIGIT_MAP = {
    'i': '1', 'l': '1', '|': '1', '!': '1',
    's': '5', 'S': '5',
    'z': '2', 'Z': '2',
    'o': '0', 'O': '0',
    'b': '6',
}

# Mapping of misread/garbled OCR unit suffixes to valid SPICE multipliers per component category
UNIT_CORRECTION_MAP = {
    'resistor': {
        'xe': 'k', 'x': 'k', 'q': 'k', 'ke': 'k', 'ko': 'k', 'kohm': 'k', 'kohms': 'k', 'k_': 'k',
        'ohm': '', 'ohms': '', 'r': '', 'omega': '',
        'mg': 'meg', 'm': 'meg', 'me': 'meg', 'megohm': 'meg',
        'g': 'g', 'ghm': 'g',
    },
    'capacitor': {
        'uu': 'u', 'uf': 'u', 'micro': 'u', 'x': 'u', 'xe': 'u', 'u_': 'u',
        'pf': 'p', 'pp': 'p',
        'nf': 'n', 'nn': 'n',
        'mf': 'm', 'm': 'm',
        'f': 'u',
    },
    'inductor': {
        'uh': 'u', 'x': 'u', 'xe': 'u',
        'nh': 'n',
        'mh': 'm', 'm': 'm',
        'h': '',
    },
    'source': {
        'v': 'v', 'volts': 'v', 'volt': 'v',
        'a': 'a', 'amps': 'a', 'amp': 'a',
    }
}


def sanitize_raw_text(raw_text: str) -> str:
    """Strip noise characters and normalize basic symbols (Greek mu, omega, whitespace)."""
    if not raw_text:
        return ""
    text = str(raw_text).strip()
    # Normalize Greek mu (μ / µ) to ASCII 'u'
    text = text.replace('\u03bc', 'u').replace('\u00b5', 'u')
    # Replace commas with dots for decimal parsing
    text = text.replace(',', '.')
    # Lowercase
    text_lower = text.lower()
    # Strip trailing ohm / omega words
    text_clean = re.sub(r'\b(ohms?|omega|\u03a9|\u2126)\b', '', text_lower)
    # Remove unwanted punctuation (except numbers, letters, dots, hyphens, plus)
    text_clean = re.sub(r'[^a-z0-9\.\-\+]', '', text_clean)
    return text_clean.strip()


def fix_handwriting_digits(numeric_str: str) -> str:
    """Apply handwriting digit corrections (e.g., 'o' -> '0', 's' -> '5') to numeric parts only."""
    res = []
    for char in numeric_str:
        if char in HANDWRITING_DIGIT_MAP:
            res.append(HANDWRITING_DIGIT_MAP[char])
        else:
            res.append(char)
    return "".join(res)


def expand_european_notation(text: str) -> str:
    """Convert European schematic notation (e.g. 4k7 -> 4.7k, 0r1 -> 0.1, 2m2 -> 2.2m)."""
    match = re.match(r'^(\d+)([krmgupn])(\d+)$', text, re.IGNORECASE)
    if match:
        num1, unit, num2 = match.groups()
        unit_lower = unit.lower()
        if unit_lower == 'r':
            return f"{num1}.{num2}"
        return f"{num1}.{num2}{unit_lower}"
    return text


def clean_leading_zeros(num_str: str) -> Optional[str]:
    """Sanitize numbers starting with 0.

    1. Pure zero (0, 0.0, 00) -> returns None (triggers category fallback default)
    2. Leading zeros on integers (045 -> 45, 010 -> 10, 01 -> 1)
    3. Decimals starting with 0 (0.1 -> .1, 0.47 -> .47)
    """
    if not num_str:
        return None

    # Check if number is numerically zero
    try:
        if float(num_str) == 0.0:
            return None
    except ValueError:
        return None

    # Handle sign
    sign = ""
    if num_str.startswith('+') or num_str.startswith('-'):
        sign = num_str[0]
        num_str = num_str[1:]

    # Case A: Decimal starting with '0.' -> replace '0.' with '.'
    if num_str.startswith('0.'):
        num_str = num_str[1:]  # '0.1' -> '.1'

    # Case B: Integer with leading zeros like '045' -> '45', '007' -> '7'
    elif num_str.startswith('0'):
        num_str = num_str.lstrip('0')
        if not num_str:
            return None

    return f"{sign}{num_str}"


def parse_numeric_and_unit(text: str) -> Tuple[Optional[str], str]:
    """Split a candidate value string into (numeric_part, raw_unit_suffix)."""
    if not text:
        return None, ""

    # First check European notation expansion
    text = expand_european_notation(text)

    # Match numeric prefix (with optional decimal point and negative sign) + rest
    match = re.match(r'^([\+\-]?[\d\.\sA-Za-z]+?)([a-zA-Z_]*)$', text)
    if not match:
        return None, text

    # Separate any leading numeric digits/letters from trailing unit
    boundary_match = re.match(r'^([\+\-]?[\d\.iIl|sSzZoObg]+)(.*)$', text)
    if boundary_match:
        digits_candidate, unit_candidate = boundary_match.groups()
        num_fixed = fix_handwriting_digits(digits_candidate)

        # Sanitize multiple decimal points (keep only first decimal)
        if num_fixed.count('.') > 1:
            parts = num_fixed.split('.')
            num_fixed = parts[0] + '.' + ''.join(parts[1:])

        # Clean leading zeros and reject pure zeros
        num_fixed = clean_leading_zeros(num_fixed)

        if num_fixed is not None:
            try:
                float(num_fixed)
                return num_fixed, unit_candidate.lower()
            except ValueError:
                pass

    return None, text.lower()


def get_component_category(comp_type: str) -> str:
    """Map canvas component type string to internal category."""
    c = str(comp_type).lower()
    if 'resistor' in c:
        return 'resistor'
    if 'capacitor' in c:
        return 'capacitor'
    if 'inductor' in c:
        return 'inductor'
    if 'source' in c or c in ('vss', 'source', 'voltage_source', 'current_source'):
        return 'source'
    return c


def is_valid_spice_value(value_str: str, comp_type: str = 'resistor') -> bool:
    """Strictly validate whether value_str is a valid SPICE literal for comp_type.
    
    Enforces rule: values CANNOT start with '0' (e.g. '045k', '010k', '0.1', '0' are invalid).
    """
    if not value_str or not isinstance(value_str, str):
        return False
    val = value_str.strip()
    if not val:
        return False

    # Enforce rule: value cannot start with '0' (or '+0', '-0')
    val_nosign = val.lstrip('+-')
    if val_nosign.startswith('0'):
        return False

    category = get_component_category(comp_type)
    # Check if value is a pure valid number
    try:
        fval = float(val)
        if fval == 0.0:
            return False
        return True
    except ValueError:
        pass

    # Check number + valid SPICE unit suffix
    match = re.match(r'^([\+\-]?\d*\.?\d+)(k|m|meg|g|u|n|p|f|v|a)?$', val, re.IGNORECASE)
    if not match:
        return False

    num_str, unit_suffix = match.groups()
    if not num_str:
        return False

    unit_suffix = (unit_suffix or '').lower()
    if category == 'resistor' and unit_suffix in ('', 'k', 'm', 'meg', 'g'):
        return True
    if category == 'capacitor' and unit_suffix in ('', 'u', 'n', 'p', 'm', 'f'):
        return True
    if category == 'inductor' and unit_suffix in ('', 'u', 'n', 'm'):
        return True
    if category == 'source' and unit_suffix in ('', 'v', 'a', 'm', 'k'):
        return True

    return True


def correct_ocr_value(raw_ocr_str: str, comp_type: str = 'resistor') -> Tuple[str, bool, str]:
    """Validate and autocorrect an OCR string into a clean SPICE component value.

    Parameters
    ----------
    raw_ocr_str : str
        Raw OCR text output (e.g. "45xe", "045k", "0.1", "10ko", "4k7", "garbage")
    comp_type : str
        Component canvas type (e.g. "resistor", "capacitor", "inductor", "source")

    Returns
    -------
    Tuple[str, bool, str]
        (corrected_value, was_originally_valid, explanation)
    """
    category = get_component_category(comp_type)
    default_val = DEFAULT_COMPONENT_VALUES.get(comp_type, DEFAULT_COMPONENT_VALUES.get(category, '1k'))

    if not raw_ocr_str:
        return default_val, False, f"Empty OCR input -> fallback default '{default_val}'"

    clean_str = sanitize_raw_text(raw_ocr_str)
    if not clean_str:
        return default_val, False, f"No valid characters in '{raw_ocr_str}' -> fallback default '{default_val}'"

    # Check if already strictly valid (must NOT start with '0')
    if is_valid_spice_value(clean_str, comp_type):
        return clean_str, True, "Value is valid SPICE literal"

    # Attempt parsing numeric + unit suffix
    num_part, unit_part = parse_numeric_and_unit(clean_str)

    if num_part is None:
        # Unable to parse any valid non-zero numeric digits -> fallback to default
        return default_val, False, f"Could not parse valid non-zero numeric component from '{raw_ocr_str}' -> default '{default_val}'"

    # Unit correction mapping
    unit_map = UNIT_CORRECTION_MAP.get(category, {})
    corrected_unit = unit_part

    if unit_part in unit_map:
        corrected_unit = unit_map[unit_part]
    elif unit_part:
        # Check if unit_part starts with a recognized unit letter (e.g. "xe" -> "k" for resistor)
        for noise_pattern, target_unit in unit_map.items():
            if unit_part.startswith(noise_pattern):
                corrected_unit = target_unit
                break
        else:
            # Category-level fallback unit if unrecognized suffix
            if category == 'resistor':
                corrected_unit = 'k' if ('k' in unit_part or 'x' in unit_part or 'e' in unit_part) else 'k'
            elif category == 'capacitor':
                corrected_unit = 'u'
            elif category == 'inductor':
                corrected_unit = 'u'
            elif category == 'source':
                corrected_unit = 'v' if 'current' not in comp_type else 'a'

    final_val = f"{num_part}{corrected_unit}"

    # Final sanity validation check
    if is_valid_spice_value(final_val, comp_type):
        explanation = f"Autocorrected '{raw_ocr_str}' -> '{final_val}' (num='{num_part}', unit='{corrected_unit}')"
        return final_val, False, explanation

    # If final_val starts with '.' like '.1k' or '.47', check if valid
    if final_val.startswith('.') and is_valid_spice_value('1' + final_val, comp_type):
        return final_val, False, f"Autocorrected '{raw_ocr_str}' -> '{final_val}' (stripped leading zero)"

    # If still invalid after correction attempts, return fallback default
    return default_val, False, f"Sanitized result '{final_val}' was invalid -> fallback default '{default_val}'"


# ═══════════════════════════════════════════════════════════════════════
# CLI SELF-TEST ENGINE
# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    test_cases = [
        ("45xe", "resistor"),
        ("045k", "resistor"),
        ("010k", "resistor"),
        ("0.1", "resistor"),
        ("0.47k", "resistor"),
        ("0", "resistor"),
        ("0.0", "resistor"),
        ("0k", "resistor"),
        ("10ko", "resistor"),
        ("4k7", "resistor"),
        ("0r1", "resistor"),
        ("100uu", "capacitor"),
        ("4.7uF", "capacitor"),
        ("10mH", "inductor"),
        ("l0k", "resistor"),
        ("s00", "resistor"),
        ("1.5.0k", "resistor"),
        ("invalid_ocr_noise", "resistor"),
        ("5V", "source"),
    ]

    print("Running OCR Validator & Autocorrect Test Suite:\n" + "=" * 60)
    for raw_input, ctype in test_cases:
        corrected, is_valid, note = correct_ocr_value(raw_input, ctype)
        status = "[Valid]" if is_valid else "[Corrected]"
        print(f"Input: {raw_input:18} | Type: {ctype:10} | {status:11} -> {corrected:10} ({note})")

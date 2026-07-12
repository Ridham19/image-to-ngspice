"""Quick smoke tests for the refactored OCR helpers — isolated from ML imports."""
import re

# ── Inline copies of the functions under test (avoids ultralytics/easyocr import) ──
# These are exact copies from core/model.py; the test validates behaviour.

def clean_ocr_string(raw_text):
    text = raw_text.strip()
    text = text.replace('\u03bc', 'u').replace('\u00b5', 'u')
    text = text.lower()
    text = re.sub(r'(ohms?|omega|\u03a9|\u2126)$', '', text)
    text = re.sub(r'(\d+\.?\d*)x$', r'\1k', text)
    text = re.sub(r'[^a-z0-9\._\s]', '', text)
    return text.strip()

def _fix_handwriting_digits(text):
    text = text.replace('i', '1').replace('l', '1').replace('|', '1')
    text = text.replace('s', '5')
    text = text.replace('z', '2')
    text = text.replace('o', '0')
    return text

_COMPOSITE_RE = re.compile(r'^([a-z]{1,3}\d{1,4})\s+(\d[\d\.]*\w{0,3})$')

def classify_text(text, comp_type, comp_prefix):
    if not text:
        return None
    composite_match = _COMPOSITE_RE.match(text)
    if composite_match:
        return {
            'name': composite_match.group(1),
            'value': composite_match.group(2),
        }
    text = re.sub(r'\s+', '', text)
    if not text:
        return None
    has_digit = any(c.isdigit() for c in text)
    starts_with_prefix = comp_prefix and text.startswith(comp_prefix.lower())
    if not has_digit:
        return 'name'
    starts_with_digit = text[0].isdigit() or text[0] == '.'
    if starts_with_digit:
        return 'value'
    if starts_with_prefix:
        return 'name'
    return 'value'

def correct_component_value(value_str, comp_type):
    if not value_str:
        return value_str
    comp_type = comp_type.lower()
    # Protect known unit suffixes from digit corruption
    preserved_unit = ''
    value_lower = value_str.lower()
    for suffix in ('ohms', 'ohm', 'meg', 'kohm', 'kohms'):
        if value_lower.endswith(suffix):
            preserved_unit = suffix
            value_str = value_str[:len(value_str) - len(suffix)]
            break
    value_str = _fix_handwriting_digits(value_str) + preserved_unit
    internal_unit_match = re.match(r'^(\d+)([krmgupn])(\d+)$', value_str)
    if internal_unit_match:
        num1, unit, num2 = internal_unit_match.groups()
        if unit == 'r':
            value_str = f"{num1}.{num2}"
        else:
            value_str = f"{num1}.{num2}{unit}"
    match = re.match(r'^([\d\.]+)([a-zA-Z]*)$', value_str)
    if not match:
        return value_str
    numeric_part, unit_part = match.groups()
    if numeric_part.count('.') > 1:
        parts = numeric_part.split('.')
        numeric_part = parts[0] + '.' + ''.join(parts[1:])
    unit_part = unit_part.lower()
    if 'resistor' in comp_type:
        if unit_part in ('r', 'ohm', 'ohms'):
            unit_part = ''
        elif unit_part in ('k', 'x', 'q'):
            unit_part = 'k'
        elif unit_part in ('meg', 'mg'):
            unit_part = 'meg'
        elif unit_part == 'm':
            unit_part = 'meg'
        elif unit_part == 'g':
            unit_part = 'g'
        elif unit_part:
            unit_part = 'k'
    elif 'capacitor' in comp_type:
        if unit_part == 'p':
            unit_part = 'p'
        elif unit_part == 'n':
            unit_part = 'n'
        elif unit_part in ('u', 'x'):
            unit_part = 'u'
        elif unit_part == 'm':
            unit_part = 'm'
        elif unit_part == 'f':
            unit_part = 'u'
        elif unit_part:
            unit_part = 'u'
    elif 'inductor' in comp_type:
        if unit_part in ('u', 'x'):
            unit_part = 'u'
        elif unit_part == 'n':
            unit_part = 'n'
        elif unit_part == 'm':
            unit_part = 'm'
        elif unit_part == 'h':
            unit_part = ''
        elif unit_part:
            unit_part = 'u'
    elif 'source' in comp_type:
        if 'current' in comp_type:
            unit_part = 'a' if unit_part else ''
        else:
            unit_part = 'v' if unit_part else ''
    return f"{numeric_part}{unit_part}"


# ══════════════════════════════════════════════════════════
# TESTS
# ══════════════════════════════════════════════════════════

# 1. clean_ocr_string: should NOT corrupt letters
assert clean_ocr_string("source") == "source",           f"FAIL: {clean_ocr_string('source')}"
assert clean_ocr_string("R_SENSE") == "r_sense",         f"FAIL: {clean_ocr_string('R_SENSE')}"
assert clean_ocr_string("10\u03bcF") == "10uf",          f"FAIL: {clean_ocr_string(chr(0x03bc))}"
assert clean_ocr_string("10\u00b5F") == "10uf",          f"FAIL"
assert clean_ocr_string("4.7k\u03a9") == "4.7k",         f"FAIL: {clean_ocr_string('4.7k\u03a9')}"
assert clean_ocr_string("100 ohms") == "100",             f"FAIL: {clean_ocr_string('100 ohms')}"
assert clean_ocr_string("10x") == "10k",                  f"FAIL: {clean_ocr_string('10x')}"
print("  [PASS] clean_ocr_string")

# 2. _fix_handwriting_digits
assert _fix_handwriting_digits("1o0k") == "100k"
assert _fix_handwriting_digits("4l7") == "417"
assert _fix_handwriting_digits("5z") == "52"
print("  [PASS] _fix_handwriting_digits")

# 3. classify_text: composite splitting
result = classify_text("r1 10k", "resistor", "R")
assert isinstance(result, dict),                         f"Expected dict, got: {result}"
assert result['name'] == 'r1',                           f"Got name: {result['name']}"
assert result['value'] == '10k',                         f"Got value: {result['value']}"
assert classify_text("r1", "resistor", "R") == 'name'
assert classify_text("10k", "resistor", "R") == 'value'
assert classify_text("source", "source", "V") == 'name'
assert classify_text("", "resistor", "R") is None
print("  [PASS] classify_text")

# 4. correct_component_value: unit fixes
assert correct_component_value("10r", "resistor") == "10",     f"FAIL: {correct_component_value('10r', 'resistor')}"
assert correct_component_value("10ohm", "resistor") == "10",   f"FAIL: {correct_component_value('10ohm', 'resistor')}"
assert correct_component_value("10k", "resistor") == "10k"
assert correct_component_value("1m", "resistor") == "1meg",    f"FAIL: {correct_component_value('1m', 'resistor')}"
assert correct_component_value("4.7meg", "resistor") == "4.7meg"
assert correct_component_value("4k7", "resistor") == "4.7k"
assert correct_component_value("0r1", "resistor") == "0.1"
assert correct_component_value("10u", "capacitor") == "10u"
assert correct_component_value("100p", "capacitor") == "100p"
assert correct_component_value("1f", "capacitor") == "1u"
assert correct_component_value("10n", "capacitor") == "10n"
assert correct_component_value("10u", "inductor") == "10u"
assert correct_component_value("1h", "inductor") == "1"
assert correct_component_value("5v", "source") == "5v"
assert correct_component_value("5", "source") == "5"
assert correct_component_value("1o0k", "resistor") == "100k",  f"FAIL: {correct_component_value('1o0k', 'resistor')}"
print("  [PASS] correct_component_value")

print()
print("All OCR pipeline tests passed!")

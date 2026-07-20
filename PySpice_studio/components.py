import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import schemdraw
from PIL import Image, ImageTk
import io
import math
import copy
from library import DB

IMG_CACHE = {}

# --- SIZES (Standardized for GRID_SIZE = 20) ---
SIZE_2_PIN = 80      
SIZE_V_SOURCE = 80   
SIZE_GND = 40        
SIZE_TRANS_H = 80    
SIZE_TRANS_W = 40    
SIZE_LABEL = 10      # <--- REDUCED from 20 to 10 for a smaller footprint

# Component Color
COMP_COLOR = '#F0F0F0' 

# --- EXACT PIN COORDINATES ---
PIN_MAP = {
    'ac_source':   [(0, -40), (0, 40)],
    'sine_source': [(0, -40), (0, 40)],
    'bjt_npn':     [(-20, 0), (20, -40), (20, 40)],
    'bjt_pnp':     [(-20, 0), (20, -40), (20, 40)],
    'capacitor':   [(-40, 0), (40, 0)],
    'current':     [(0, -40), (0, 40)],
    'diode':       [(-40, 0), (40, 0)],
    'gnd':         [(0, -20)],
    'inductor':    [(-40, 0), (40, 0)],
    'label':       [(0, 0)], # Center anchored
    'junction':    [(0, 0)],
    'pulse':       [(0, -40), (0, 40)],
    'resistor':    [(-40, 0), (40, 0)],
    'source':      [(0, -40), (0, 40)],
    # New additions
    'capacitor_polarized': [(-40, 0), (40, 0)],
    'resistor_photo':       [(-40, 0), (40, 0)],
    'diode_led':            [(-40, 0), (40, 0)],
    'diode_zener':          [(-40, 0), (40, 0)],
    'phototransistor':      [(-20, 0), (20, -40), (20, 40)],
    'opamp':                [(-30, -20), (-30, 20), (30, 0)],
    'transformer':          [(-40, -20), (-40, 20), (40, -20), (40, 20)],
    'nmos':                 [(-20, 0), (20, -40), (20, 40)],
    'pmos':                 [(-20, 0), (20, -40), (20, 40)],
    'vss':                  [(0, -40), (0, 40)],
    'current_source':       [(0, -40), (0, 40)],
    'pulse_source':         [(0, -40), (0, 40)],
    'sine_source_td':       [(0, -40), (0, 40)],
    'exp_source':           [(0, -40), (0, 40)],
    'pwl_source':           [(0, -40), (0, 40)],
    'sffm_source':          [(0, -40), (0, 40)],
    'am_source':            [(0, -40), (0, 40)],
}

class ComponentHelper:
    @staticmethod
    def render_image(c_type, rotation=0, zoom_scale=1.0, comp=None):
        if c_type == 'wire': return None
        if c_type not in DB: return ComponentHelper.create_fallback_image(c_type)

        zoom_key = round(zoom_scale, 1)
        num_pins = int(comp.params.get('num_pins', '2')) if (comp and c_type == 'ic') else 2
        key = (c_type, rotation, zoom_key, num_pins)
        if key in IMG_CACHE: return IMG_CACHE[key]

        if c_type == 'ic':
            try:
                half = (num_pins + 1) // 2
                body_h = max(30, half * 20)
                img_w = int(100 * zoom_scale)
                img_h = int((body_h + 20) * zoom_scale)
                img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
                
                from PIL import ImageDraw
                draw = ImageDraw.Draw(img)
                
                scale = zoom_scale
                bx1, by1 = int(20 * scale), int(10 * scale)
                bx2, by2 = int(80 * scale), int((body_h + 10) * scale)
                
                draw.rectangle([bx1, by1, bx2, by2], fill="#333333", outline=COMP_COLOR, width=int(2 * scale))
                
                start_y = - (half - 1) * 10
                for i in range(half):
                    y = int(((body_h + 20) / 2 + start_y + i * 20) * scale)
                    draw.line([int(10 * scale), y, bx1, y], fill=COMP_COLOR, width=int(1.5 * scale))
                    
                for i in range(num_pins - half):
                    y = int(((body_h + 20) / 2 + start_y + (num_pins - half - 1 - i) * 20) * scale)
                    draw.line([bx2, y, int(90 * scale), y], fill=COMP_COLOR, width=int(1.5 * scale))
                    
                if rotation != 0: img = img.rotate(-rotation, expand=True)
                
                tk_img = ImageTk.PhotoImage(img)
                IMG_CACHE[key] = (tk_img, img)
                return (tk_img, img)
            except:
                return ComponentHelper.create_fallback_image(c_type)

        try:
            data = DB[c_type]
            e = data['element'](color=COMP_COLOR)
            d = schemdraw.Drawing(show=False)
            d.add(e)
            
            dpi = int(120 * max(0.5, zoom_scale))
            buf = io.BytesIO()
            d.save(buf, transparent=True, dpi=dpi)
            buf.seek(0)
            img = Image.open(buf)
            bbox = img.getbbox()
            if bbox: img = img.crop(bbox)
            
            w, h = img.size
            shape = data['shape']
            
            # --- UPDATED LABEL SCALING ---
            if shape == 'label':
                target_sz = int(SIZE_LABEL * zoom_scale)
                img = img.resize((target_sz, target_sz), Image.Resampling.LANCZOS)
            
            elif shape in ['3_pin_bjt', '3_pin_fet']:
                img = img.resize((int(SIZE_TRANS_W * zoom_scale), int(SIZE_TRANS_H * zoom_scale)), Image.Resampling.LANCZOS)

            elif shape == 'v_source':
                target_h = int(SIZE_V_SOURCE * zoom_scale)
                ratio = target_h / float(h)
                img = img.resize((int(w * ratio), target_h), Image.Resampling.LANCZOS)

            elif shape == '1_pin':
                target_h = int(SIZE_GND * zoom_scale)
                ratio = target_h / float(h)
                img = img.resize((int(w * ratio), target_h), Image.Resampling.LANCZOS)

            elif shape == 'opamp':
                img = img.resize((int(70 * zoom_scale), int(60 * zoom_scale)), Image.Resampling.LANCZOS)

            elif shape == 'transformer':
                img = img.resize((int(80 * zoom_scale), int(60 * zoom_scale)), Image.Resampling.LANCZOS)

            else: # Standard 2-pin
                target_len = int(SIZE_2_PIN * zoom_scale)
                ratio = target_len / float(w)
                img = img.resize((target_len, int(h * ratio)), Image.Resampling.LANCZOS)
            
            # Center on Tile
            dim = max(img.size) + 10
            final = Image.new("RGBA", (dim, dim), (0,0,0,0))
            final.paste(img, ((dim-img.size[0])//2, (dim-img.size[1])//2))
            img = final

            if rotation != 0: img = img.rotate(-rotation, expand=False)
            
            tk_img = ImageTk.PhotoImage(img)
            IMG_CACHE[key] = (tk_img, img)
            return (tk_img, img)
        except:
            return ComponentHelper.create_fallback_image(c_type)

    @staticmethod
    def create_fallback_image(c_type):
        img = Image.new('RGBA', (40, 40), (0,0,0,0))
        tk_img = ImageTk.PhotoImage(img)
        return (tk_img, img)

class Component:
    def __init__(self, c_type, x, y, name):
        self.type = c_type
        self.x = x; self.y = y
        self.name = name
        self.rotation = 0
        
        if c_type in DB:
            self.params = copy.deepcopy(DB[c_type]['params'])
            self.shape_type = DB[c_type]['shape']
        else:
            self.params = {'value': '1k'}
            self.shape_type = '2_pin'

        self.canvas_id = None; self.text_id = None; self.pin_ids = []; self.img_ref = None 

    @property
    def value(self):
        if self.type == 'label': return self.params.get('name', 'NET')
        k = list(self.params.keys())
        return self.params[k[0]] if k else ""
        
    @value.setter
    def value(self, v):
        if self.type == 'label': self.params['name'] = v
        else:
            k = list(self.params.keys())
            if k: self.params[k[0]] = v

    def rotate(self): self.rotation = (self.rotation + 90) % 360

    def get_pins(self):
        rad = math.radians(self.rotation)
        c, s = int(math.cos(rad)), int(math.sin(rad))
        def rot(pt):
            px, py = pt
            nx = px * c - py * s
            ny = px * s + py * c
            return (self.x + nx, self.y + ny)
        
        def snap(pt): return (round(pt[0]/20)*20, round(pt[1]/20)*20)

        if self.type == 'ic':
            num_pins = int(self.params.get('num_pins', '2'))
            half = (num_pins + 1) // 2
            left_pins = []
            right_pins = []
            start_y = - (half - 1) * 10
            for i in range(half):
                y = start_y + i * 20
                left_pins.append((-40, y))
            for i in range(num_pins - half):
                y = start_y + (num_pins - half - 1 - i) * 20
                right_pins.append((40, y))
            pins_list = left_pins + right_pins
            return [snap(rot(p)) for p in pins_list]

        if self.type in PIN_MAP:
            return [snap(rot(p)) for p in PIN_MAP[self.type]]

        d = 40
        return [snap(rot((-d, 0))), snap(rot((d, 0)))]
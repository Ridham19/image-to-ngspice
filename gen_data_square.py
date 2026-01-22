import schemdraw
import schemdraw.elements as elm
import matplotlib.pyplot as plt
import random
import os
import shutil

# --- CONFIGURATION ---
TRAIN_DIR = "dataset_training_annotated"
TEST_DIR = "DATA/dataset_testing_clean"
NUM_IMAGES = 50

# --- GRID CONSTANTS ---
# We define the exact "physical" locations of our circuit nodes
TOP_Y = 6.0
BOT_Y = 0.0
START_X = 0.0
BRANCH_SPACING = 5.0
BOX_WIDTH = 2.5 

LOAD_COMPONENTS = [
    (elm.Resistor, 'R', 'RESISTOR', False),
    (elm.Capacitor, 'C', 'CAPACITOR', False),
    (elm.Inductor, 'L', 'INDUCTOR', False),
    (elm.Diode, 'D', 'DIODE', False),
    (elm.LED, 'LED', 'LED', False),
    (elm.BjtNpn, 'Q', 'BJT', True),
    (elm.NFet, 'M', 'MOSFET', True)
]

SOURCES = [
    (elm.SourceV, 'V', 'VOLTAGE'),
    (elm.SourceSin, 'Vac', 'AC_SOURCE'),
    (elm.Battery, 'Bat', 'BATTERY')
]

def draw_exact_box(d, x, y_top, y_bot, label, color='green'):
    """
    Draws a box using absolute grid coordinates.
    """
    x1 = x - BOX_WIDTH/2
    x2 = x + BOX_WIDTH/2
    
    # Standard components go full height
    y1 = y_bot + 0.2
    y2 = y_top - 0.2
    
    # Transistors are short, sitting at the top
    if label in ['BJT', 'MOSFET']:
        y1 = y_top - 3.0 # Only box the top half
    
    # Draw Box
    box_color = 'yellow' if label == 'JUNCTION' else color
    d += elm.Rect(p1=(x1, y1), p2=(x2, y2)).color(box_color).linestyle('--').linewidth(2)
    
    # Draw Text (Centered above)
    d += elm.Label(label=label, size=10, color='red').at((x, y2 + 0.6))

def draw_circuit_variant(seed, save_path, mode='clean'):
    random.seed(seed)
    
    with schemdraw.Drawing(show=False) as d:
        d.config(unit=3.0, lw=2.5)
        
        # --- 1. DRAW SOURCE (Fixed at X=0) ---
        S_Class, s_pre, s_name = random.choice(SOURCES)
        
        # Manually draw from Bot to Top
        source = S_Class().at((START_X, BOT_Y)).to((START_X, TOP_Y))
        if mode == 'clean': source.label(label=f"{s_pre}1")
        d += source
        
        if mode == 'annotated':
            draw_exact_box(d, START_X, TOP_Y, BOT_Y, s_name)

        # --- 2. DRAW BRANCHES ---
        num_branches = random.randint(2, 4)
        current_x = START_X
        
        for i in range(num_branches):
            # Calculate next grid point
            next_x = current_x + BRANCH_SPACING
            
            # A. Manually Draw Rails (Wires)
            d += elm.Line().at((current_x, TOP_Y)).to((next_x, TOP_Y)) # Top Wire
            d += elm.Line().at((current_x, BOT_Y)).to((next_x, BOT_Y)) # Bot Wire
            
            # B. Draw Junction Dots
            d += elm.Dot().at((next_x, TOP_Y))
            if mode == 'annotated': 
                # Small fixed box for junction
                d += elm.Rect(p1=(next_x-0.4, TOP_Y-0.4), p2=(next_x+0.4, TOP_Y+0.4)).color('yellow').linestyle('--').linewidth(2)
                d += elm.Label(label='JUNCTION', size=9, color='red').at((next_x, TOP_Y+0.6))
            
            d += elm.Dot().at((next_x, BOT_Y))
            if mode == 'annotated':
                d += elm.Rect(p1=(next_x-0.4, BOT_Y-0.4), p2=(next_x+0.4, BOT_Y+0.4)).color('yellow').linestyle('--').linewidth(2)
                d += elm.Label(label='JUNCTION', size=9, color='red').at((next_x, BOT_Y-0.8))

            # C. Draw Vertical Component
            CompClass, prefix, name, is_3term = random.choice(LOAD_COMPONENTS)
            
            if is_3term:
                # Transistor: Place at top, manually wire bottom
                # Note: We use .down() to orient it, but place it at the exact grid point
                branch = CompClass().at((next_x, TOP_Y)).down()
                if mode == 'clean': branch.label(label=f"{prefix}{i+1}")
                d += branch
                
                # Manual Wire to Bottom Rail
                # Connect from Emitter/Source down to (next_x, BOT_Y)
                leg_pos = branch.emitter if name == 'BJT' else branch.source
                d += elm.Line().at(leg_pos).to((next_x, BOT_Y))
                
                # Manual Stub for Base/Gate
                side_pos = branch.base if name == 'BJT' else branch.gate
                d += elm.Line().at(side_pos).left().length(1.0)
                
            else:
                # 2-Terminal: Connect strictly Top to Bot
                branch = CompClass().at((next_x, TOP_Y)).to((next_x, BOT_Y))
                if mode == 'clean': branch.label(label=f"{prefix}{i+1}")
                d += branch

            # D. Draw Component Box (Annotated Mode)
            if mode == 'annotated':
                draw_exact_box(d, next_x, TOP_Y, BOT_Y, name)
            
            current_x = next_x

        # --- 3. DRAW GROUND ---
        d += elm.Ground().at((START_X, BOT_Y))
        if mode == 'annotated':
            # Fixed Ground Box
            d += elm.Rect(p1=(START_X-0.6, BOT_Y-1.2), p2=(START_X+0.6, BOT_Y)).color('green').linestyle('--').linewidth(2)
            d += elm.Label(label='GROUND', size=9, color='red').at((START_X, BOT_Y-1.5))

        d.save(save_path, transparent=False, dpi=100)
    
    plt.close('all')

def main():
    for f in [TRAIN_DIR, TEST_DIR]:
        if os.path.exists(f): shutil.rmtree(f)
        os.makedirs(f)
    
    print(f"Generating {NUM_IMAGES} manual-grid circuits...")
    for i in range(NUM_IMAGES):
        seed = i * 3030
        try:
            draw_circuit_variant(seed, os.path.join(TEST_DIR, f"circuit_{i}.png"), mode='clean')
            draw_circuit_variant(seed, os.path.join(TRAIN_DIR, f"circuit_{i}.png"), mode='annotated')
            if i % 10 == 0: print(f"  .. {i}")
        except Exception as e:
            print(f"Error {i}: {e}")

    print("Generation Complete.")

if __name__ == "__main__":
    main()
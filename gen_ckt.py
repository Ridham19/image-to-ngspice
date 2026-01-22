import schemdraw
import schemdraw.elements as elm
import random
import os

# Create output directory
OUTPUT_DIR = "DATA/dataset_closed_loop"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_random_component(label_idx, orientation='right'):
    """Returns a random component with a label."""
    # List of available component classes (excluding SourceV which is fixed)
    options = [
        (elm.Resistor, 'R'),
        (elm.Capacitor, 'C'),
        (elm.Inductor, 'L'),
        # We can add BJTs/MOSFETs as 2-terminal "diodes" or switches for simple loop gen, 
        # but for closed-loop connectivity in simple logic, 2-terminal is safest.
        # We will wrap standard BJT/MOSFET to fit in a 2-node slot if selected.
        (elm.Diode, 'D'), 
    ]
    
    CompClass, prefix = random.choice(options)
    label = f"{prefix}{label_idx}"
    
    comp = CompClass().label(label)
    
    if orientation == 'right': return comp.right()
    elif orientation == 'down': return comp.down()
    elif orientation == 'left': return comp.left()
    elif orientation == 'up': return comp.up()
    return comp

def generate_circuit(num_components, filename):
    """
    Generates a closed-loop circuit with exactly 'num_components'.
    Strategy: Ladder Network / Mesh.
    1. Always starts with SourceV (1 component).
    2. Remaining (N-1) components are distributed into 'slots' of a grid.
    """
    if num_components < 2:
        print("Error: Minimum 2 components required for a loop.")
        return

    # We need to place N-1 components (since V1 is always there)
    comps_to_place = num_components - 1
    
    # Define slots in a single rectangular loop: [Top, Right, Bottom]
    # For higher counts, we add a "Middle Vertical" rung (Parallel branch)
    # Slots: 'top1', 'mid_vert', 'top2', 'right', 'bot2', 'bot1'
    
    # Structure Plan based on Count:
    # 2 comps: Source + (Top/Right/Bot) -> Simple Loop
    # 3-8 comps: Source + Mesh with random cross-bars
    
    with schemdraw.Drawing(show=False) as d:
        d.config(unit=3, lw=2.0)
        
        # 1. Fixed Source (Left side)
        d += elm.SourceV().up().label('V1')
        top_node = d.here
        
        # 2. Determine Topology
        # We model a ladder with 1 or 2 loops depending on component count
        has_mid_branch = random.choice([True, False]) if num_components >= 4 else False
        
        # Allocate components to segments
        # Segments: Top-Left, Mid-Branch, Top-Right, Right-Drop, Bot-Right, Bot-Left
        segments = {
            'top_L': 0, 'mid_vert': 0, 'top_R': 0, 
            'right_vert': 0, 'bot_R': 0, 'bot_L': 0
        }
        
        # Essential segments (must exist as wires at least)
        keys = ['top_L', 'right_vert', 'bot_L']
        if has_mid_branch:
            keys += ['mid_vert', 'top_R', 'bot_R']
            
        # Randomly distribute component counts into these slots
        while comps_to_place > 0:
            target = random.choice(keys)
            segments[target] += 1
            comps_to_place -= 1
            
        # --- DRAWING ---
        
        # SEGMENT: Top Left
        d.add(elm.Line().right().length(1)) # small spacer
        if segments['top_L'] > 0:
            for _ in range(segments['top_L']):
                d += get_random_component(random.randint(1,99), 'right')
        else:
            d += elm.Line().right().length(3)
            
        mid_top_node = d.here # Save for mid branch
        
        if has_mid_branch:
            # SEGMENT: Top Right
            if segments['top_R'] > 0:
                for _ in range(segments['top_R']):
                    d += get_random_component(random.randint(1,99), 'right')
            else:
                d += elm.Line().right().length(3)
        
        # SEGMENT: Right Vertical (The far right drop)
        right_top_node = d.here
        if segments['right_vert'] > 0:
            for _ in range(segments['right_vert']):
                d += get_random_component(random.randint(1,99), 'down')
        else:
            d += elm.Line().down().length(3)
            
        # SEGMENT: Bottom Right
        if has_mid_branch:
            if segments['bot_R'] > 0:
                for _ in range(segments['bot_R']):
                    d += get_random_component(random.randint(1,99), 'left')
            else:
                d += elm.Line().left().length(3)
        
        mid_bot_node = d.here # Save for mid branch connection

        # SEGMENT: Middle Vertical (The parallel branch)
        if has_mid_branch:
            d.push()
            d.here = mid_top_node # Go back to top
            # Draw down to mid_bot_node
            if segments['mid_vert'] > 0:
                 for _ in range(segments['mid_vert']):
                    d += get_random_component(random.randint(1,99), 'down')
            else:
                d += elm.Line().down().tox(mid_bot_node)
            d.pop()
            
            # Ensure connection at bottom if sizes varied
            d.here = mid_bot_node

        # SEGMENT: Bottom Left (Return to source)
        if segments['bot_L'] > 0:
            for _ in range(segments['bot_L']):
                d += get_random_component(random.randint(1,99), 'left')
        else:
            d += elm.Line().left().tox(top_node[0]) # Connect back to V1 x-coord

        # Close the loop completely (connect to V1 bottom)
        d += elm.Line().to((top_node[0], 0)) # Assuming V1 started at 0,0 and went up
        d += elm.Ground()

        # Save
        d.save(filename, dpi=150)
        print(f"Generated {filename} with {num_components} components.")

def main():
    # Counts requested: 2, 3, 4, 5, 6, 7, 8
    counts = [2, 3, 4, 5, 6, 7, 8]
    images_per_count = 10
    
    for count in counts:
        print(f"--- Generating circuits with {count} components ---")
        for i in range(images_per_count):
            fname = os.path.join(OUTPUT_DIR, f"closed_loop_{count}comps_{i}.png")
            try:
                generate_circuit(count, fname)
            except Exception as e:
                print(f"Retry {i} due to error: {e}")
                # Simple retry logic
                generate_circuit(count, fname)

if __name__ == "__main__":
    main()
import os
import schemdraw
import schemdraw.elements as elm
import matplotlib.pyplot as plt
import inspect
import gc

# --- Configuration ---
OUTPUT_DIR = "dataset_clean_v1"
DIRECTIONS = ['up', 'down', 'left', 'right']
UNIT_SIZE = 3.0
LINE_WIDTH = 2.0

def get_schemdraw_classes():
    """
    Dynamically finds all valid component classes in schemdraw.elements.
    """
    valid_classes = []
    # inspect.getmembers returns (name, value) tuples
    for name, obj in inspect.getmembers(elm):
        if inspect.isclass(obj) and issubclass(obj, elm.Element) and obj is not elm.Element:
            # Filter out private classes (start with _) or base classes usually not drawn alone
            if not name.startswith('_'):
                valid_classes.append((name, obj))
    return valid_classes

def generate_component_images():
    # 1. Create Root Directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Disable pop-up windows to save RAM
    plt.ioff()
    
    # 2. Find all components
    classes = get_schemdraw_classes()
    print(f"Found {len(classes)} component types in schemdraw library.")
    print("Starting generation... (Press Ctrl+C to stop)")

    successful_counts = 0
    
    # 3. Iterate through every component class
    for idx, (comp_name, CompClass) in enumerate(classes):
        
        # Create a folder for this specific component
        comp_folder = os.path.join(OUTPUT_DIR, comp_name)
        os.makedirs(comp_folder, exist_ok=True)
        
        print(f"Processing [{idx+1}/{len(classes)}]: {comp_name}")
        
        # Try to draw in all 4 directions
        for direction in DIRECTIONS:
            save_path = os.path.join(comp_folder, f"{comp_name}_{direction}.png")
            
            try:
                with schemdraw.Drawing(show=False) as d:
                    d.config(unit=UNIT_SIZE, lw=LINE_WIDTH)
                    
                    # Instantiate component
                    # Some complex components require args; we wrap in try-except
                    try:
                        element = CompClass()
                    except TypeError:
                        # Skip components that REQUIRE arguments (like some logic gates needing inputs)
                        continue

                    # Apply direction
                    if direction == 'up': element.up()
                    elif direction == 'down': element.down()
                    elif direction == 'left': element.left()
                    else: element.right()
                    
                    d.add(element)
                    d.save(save_path, dpi=100)
                    successful_counts += 1
                    
            except Exception as e:
                # If a specific component crashes schemdraw, just skip it and move on
                pass
            finally:
                # CRITICAL: Close plot to prevent memory leak
                plt.close('all')

        # Force garbage collection every 10 components to keep laptop cool
        if idx % 10 == 0:
            gc.collect()

    print(f"\nCompleted! Generated {successful_counts} images in '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    generate_component_images()
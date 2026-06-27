#/usr/bin/env python3
"""
test_wire_detection.py

A standalone test script to visualize and evaluate the wire detection and tracing performance 
of the ML backend pipeline.

It allows you to select any circuit image from your PC and displays:
1. Component & Pin Detections (YOLO + OCR + Anchor Calibration)
2. Isolated Wire Mask (Collision masking blanking component interiors)
3. Traced Connection Graph (DFS pixel traversal routing, color-coded node clusters)

Usage:
    python test_wire_detection.py [path/to/image.png]
"""

import os
import sys
import math
import random
import cv2
import numpy as np

# Ensure the WebD backend folder is in the system path for imports
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
WEBD_BACKEND_PATH = os.path.join(WORKSPACE_ROOT, "WebD", "backend")
if WEBD_BACKEND_PATH not in sys.path:
    sys.path.insert(0, WEBD_BACKEND_PATH)

try:
    from WebD.backend.core.model import ComponentDetector
    from WebD.backend.core.processing import preprocess_image, separate_layers, compute_pin_anchors
    from WebD.backend.core.netlist import trace_nodes
    print("✅ Successfully imported WebD core modules.")
except ImportError as e:
    try:
        from core.model import ComponentDetector
        from core.processing import preprocess_image, separate_layers, compute_pin_anchors
        from core.netlist import trace_nodes
        print("✅ Successfully imported WebD core modules (via core fallback).")
    except ImportError:
        print(f"❌ Error importing core modules: {e}")
        print("Ensure the folder 'WebD/backend/' and its contents are present.")
        sys.exit(1)


# Define a color palette for visualization (BGR format for OpenCV)
def get_node_color(node_id):
    """Return a unique, vibrant color for each node ID."""
    if node_id == "0":
        return (0, 255, 0)  # Green for Ground / Reference node
    
    # Predefined high-contrast BGR colors
    colors = [
        (255, 0, 255),   # Magenta
        (0, 165, 255),   # Orange
        (255, 255, 0),   # Cyan
        (0, 255, 255),   # Yellow
        (255, 0, 0),     # Blue
        (0, 0, 255),     # Red
        (128, 0, 128),   # Purple
        (0, 128, 128),   # Teal
        (203, 192, 255), # Pink
        (255, 128, 0),   # Light Blue/Azure
        (128, 255, 0),   # Lime Green
    ]
    try:
        idx = int(node_id) - 1
        return colors[idx % len(colors)]
    except ValueError:
        # Fallback for NC or custom node names
        return (120, 120, 120)


def add_panel_header(img, title_text):
    """Add a clean header band to the top of an image panel."""
    h, w = img.shape[:2]
    header_h = 50
    # Create black header block
    header = np.zeros((header_h, w, 3), dtype=np.uint8)
    # Add text
    cv2.putText(
        header, title_text, (15, 32), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA
    )
    # Concatenate vertically
    return np.vstack((header, img))


def resize_to_height(img, target_height):
    """Resize image to target height while maintaining aspect ratio."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img
    target_width = int(w * (target_height / h))
    return cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test and visualize wire detection pipeline.")
    parser.add_argument("image_path", nargs="?", help="Path to input image.")
    parser.add_argument("--no-gui", action="store_true", help="Do not display the GUI window (only save on disk).")
    args = parser.parse_args()

    # ── 1. Determine Input Image Path ──
    image_path = args.image_path
    if not image_path:
        # Prompt user with file dialog so they can select any image on their PC
        print("📁 Opening file dialog to choose an image...")
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()  # Hide main window
            image_path = filedialog.askopenfilename(
                title="Select Circuit Image for Wire Detection Test",
                filetypes=[
                    ("Image Files", "*.png *.jpg *.jpeg *.webp *.bmp *.JPG"),
                    ("All Files", "*.*")
                ]
            )
            root.destroy()
        except Exception as e:
            print(f"⚠️ Could not open file selection dialog: {e}")

    # Fallback to random test images if no path chosen
    if not image_path or not os.path.exists(image_path):
        print("⚠️ No valid image path provided. Checking default test directory...")
        fallback_dir = os.path.join(WORKSPACE_ROOT, "random_images_for_test")
        if os.path.exists(fallback_dir):
            images = [f for f in os.listdir(fallback_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            if images:
                image_path = os.path.join(fallback_dir, images[0])
                print(f"💡 Defaulted to fallback test image: {image_path}")
            else:
                print("❌ No fallback test images found in 'random_images_for_test/' directory.")
                sys.exit(1)
        else:
            print("❌ File dialog cancelled and no 'random_images_for_test/' directory exists.")
            sys.exit(1)

    print(f"📷 Processing Image: {image_path}")

    # ── 2. Run YOLO + EasyOCR Detection ──
    # Instantiate ComponentDetector using weights in WebD/weights/best.pt
    # Passing "../weights/best.pt" matches what WebD/backend uses
    print("🤖 Initializing Component Detector model...")
    detector = ComponentDetector(model_name="../weights/best.pt")
    
    print("🔍 Running YOLO + EasyOCR detection pipeline...")
    # Run detector and save results to a temporary JSON in root
    temp_json_path = os.path.join(WORKSPACE_ROOT, "temp_detected_components.json")
    detected_comps = detector.detect(image_path, output_file=temp_json_path)
    
    print(f"✅ Detection complete! Found {len(detected_comps)} components.")

    # ── 3. Image Preprocessing & Layer Separation ──
    print("🧹 Preprocessing image filters...")
    original, gray, binary = preprocess_image(image_path)
    
    print("✂️ Masking component interiors to isolate wire lines...")
    component_mask, wire_mask, debug_overlay = separate_layers(
        gray, binary, detections=detected_comps
    )

    # ── 4. DFS Graph Traversal & Wire Tracing ──
    print("🔌 Traversing wire pixel graph (DFS tracing)...")
    connections = trace_nodes(wire_mask, detected_comps)
    print(f"✅ Wire tracing complete! Found {len(connections)} node connections.")

    # ── 5. Generate Vis Panels ──
    h_orig, w_orig = original.shape[:2]

    # Panel 1: Component & Pin Detections
    panel1 = original.copy()
    
    # Draw YOLO boxes and names
    for comp in detected_comps:
        comp_type = comp.get("type", "")
        if comp_type in ["wire", "junction"]:
            continue
            
        box = comp.get("box")
        if box:
            bx, by, bw, bh = map(int, box)
            # Draw bounding box
            cv2.rectangle(panel1, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            
            # Label
            name = comp.get("name", "U")
            val = comp.get("value", "")
            label_str = f"{name} ({comp_type})"
            if val and val != "TEXT_FOUND":
                label_str += f" = {val}"
                
            cv2.putText(
                panel1, label_str, (bx, max(15, by - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA
            )
            
            # If there's an associated text box, draw a faint orange border around it
            if "text_box" in comp:
                tx, ty, tw, th = map(int, comp["text_box"])
                cv2.rectangle(panel1, (tx, ty), (tx + tw, ty + th), (0, 140, 255), 1)

    # Draw pin anchors
    pin_anchors = compute_pin_anchors(detected_comps)
    for pin in pin_anchors:
        cv2.circle(panel1, (pin["x"], pin["y"]), 5, (0, 0, 255), -1)  # Red filled dot
        # Tiny label near pin
        cv2.putText(
            panel1, f"p{pin['pin_id']}", (pin["x"] + 6, pin["y"] + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 220), 1, cv2.LINE_AA
        )

    # Panel 2: Raw Edge Detection Mask
    if len(binary.shape) == 2:
        panel2 = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    else:
        panel2 = binary.copy()

    # Panel 3: Isolated Wire Mask (using backend's pre-rendered debug overlay)
    # Ensure it's in BGR format
    if len(debug_overlay.shape) == 2:
        panel3 = cv2.cvtColor(debug_overlay, cv2.COLOR_GRAY2BGR)
    else:
        panel3 = debug_overlay.copy()

    # Panel 4: Traced Connection Graph
    # Use a dimmed version of original as background to make colored traces stand out
    panel4 = original.copy()
    panel4 = cv2.addWeighted(panel4, 0.35, np.zeros_like(panel4), 0.65, 0)
    
    # Draw connections (traced wire paths)
    for conn in connections:
        node_id = conn.get("node", "NC")
        pts = conn.get("points", [])
        if not pts:
            continue
            
        color = get_node_color(node_id)
        
        # Draw the simplified polyline segments
        for i in range(len(pts) - 1):
            p1 = (int(pts[i]["x"]), int(pts[i]["y"]))
            p2 = (int(pts[i+1]["x"]), int(pts[i+1]["y"]))
            cv2.line(panel4, p1, p2, color, 3, cv2.LINE_AA)
            
        # Draw node labels on wire endpoints for readability
        mid_pt_idx = len(pts) // 2
        label_pos = (int(pts[mid_pt_idx]["x"]), int(pts[mid_pt_idx]["y"]))
        cv2.rectangle(
            panel4, 
            (label_pos[0] - 10, label_pos[1] - 10), 
            (label_pos[0] + 10, label_pos[1] + 10), 
            (40, 40, 40), -1
        )
        cv2.putText(
            panel4, node_id, (label_pos[0] - 5, label_pos[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
        )

    # Draw color-coded pin nodes to show matching assignments
    for comp in detected_comps:
        comp_type = comp.get("type", "")
        if comp_type in ["wire", "junction"]:
            continue
            
        comp_nodes = comp.get("nodes", [])
        comp_pins = compute_pin_anchors([comp])
        
        for pin, node_id in zip(comp_pins, comp_nodes):
            if node_id == "NC":
                cv2.circle(panel4, (pin["x"], pin["y"]), 6, (120, 120, 120), -1)  # Gray for NC
            else:
                color = get_node_color(node_id)
                cv2.circle(panel4, (pin["x"], pin["y"]), 6, color, -1)  # Color matching node
                cv2.circle(panel4, (pin["x"], pin["y"]), 7, (255, 255, 255), 1)  # White outline
                # Node index text
                cv2.putText(
                    panel4, node_id, (pin["x"] + 8, pin["y"] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
                )

    # ── 6. Assemble Final Side-by-Side Image ──
    # Resize panels to a uniform height for side-by-side display
    target_h = 600
    p1_res = resize_to_height(panel1, target_h)
    p2_res = resize_to_height(panel2, target_h)
    p3_res = resize_to_height(panel3, target_h)
    p4_res = resize_to_height(panel4, target_h)
    
    # Add title headers to each panel
    p1_final = add_panel_header(p1_res, "1. Component & Pin Detections")
    p2_final = add_panel_header(p2_res, "2. Raw Edge Detection Mask")
    p3_final = add_panel_header(p3_res, "3. Isolated Wire Mask")
    p4_final = add_panel_header(p4_res, "4. Traced Connection Graph")
    
    # Stack panels horizontally
    combined_vis = np.hstack((p1_final, p2_final, p3_final, p4_final))
    
    # Save the resulting visualization image
    output_path = os.path.join(WORKSPACE_ROOT, "wire_detection_test_result.png")
    cv2.imwrite(output_path, combined_vis)
    print(f"💾 Visual report saved to: {output_path}")

    # Display the result to the user if not running with --no-gui
    if not args.no_gui:
        print("📺 Displaying visual window...")
        try:
            import tkinter as tk
            from PIL import Image, ImageTk
            
            root = tk.Tk()
            root.title("Wire Detection Test Suite — Visual Evaluation")
            root.configure(bg="#2d2d2d")
            
            # Load the saved image using PIL
            pil_img = Image.open(output_path)
            
            # Scale down if it exceeds a reasonable screen width
            screen_w = 1200
            w_img, h_img = pil_img.size
            if w_img > screen_w:
                scale = screen_w / w_img
                pil_img = pil_img.resize((screen_w, int(h_img * scale)), Image.Resampling.LANCZOS)
                
            tk_img = ImageTk.PhotoImage(pil_img)
            lbl = tk.Label(root, image=tk_img, bg="#2d2d2d")
            lbl.image = tk_img
            lbl.pack(padx=10, pady=10)
            
            btn = tk.Button(
                root, text="Close Visualizer", command=root.destroy,
                bg="#0078D7", fg="white", font=("Segoe UI", 10, "bold"),
                relief="flat", padx=25, pady=8
            )
            btn.pack(pady=(0, 15))
            
            root.mainloop()
        except Exception as e:
            print(f"⚠️ Could not display visual window via Tkinter: {e}")
            print(f"Please view the output file manually at: {output_path}")
    else:
        print("⏭️ Skipping visual window display (--no-gui active).")
    print("👋 Shutting down.")


if __name__ == "__main__":
    main()

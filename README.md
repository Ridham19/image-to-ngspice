# ⚡ Image-to-NGSPICE (Handwritten Circuit to Simulation)

> 🚧 **WORK IN PROGRESS:** This project is currently under active development. The codebase, pipeline, and features are subject to rapid changes and restructuring. 🚧

## 📖 Overview
This project aims to completely bridge the gap between physical circuit sketches and simulation software. It is an end-to-end Machine Learning pipeline designed to take an image of a handwritten circuit diagram, detect the electronic components, trace the wire routing, and automatically generate a fully executable **NGSPICE netlist**.

## ✨ Core Pipeline
1. **Object Detection:** Utilizes a custom-trained YOLO model to identify hand-drawn components (Resistors, Capacitors, Diodes, Voltage Sources, Ground, etc.).
2. **Wire Tracing & Node Extraction:** Leverages `OpenCV` to isolate the wire masks, extract spatial relationships, and determine topological nodes using center-of-mass sorting and component-core slicing to prevent short-circuits.
3. **Netlist Generation:** Maps the detected components and their corresponding nodes to standard SPICE syntax, automatically assigning the ground node to `0`.

## 🗂️ Project Structure
The primary, actively developed codebase is located in the `proper/` directory:
* `proper/main.py`: The main entry point for the application.
* `proper/modules/model.py`: Handles YOLO model loading and inference.
* `proper/modules/processing.py`: Image preprocessing and OpenCV wire masking.
* `proper/modules/netlist.py`: Graph extraction, node mapping, and SPICE code generation.
* `proper/modules/labeling.py`: Custom GUI tool for rich-annotating components, wires, and nodes.

*(Note: The root directory contains various deprecated scripts, data generation testing scripts, and dataset folders used during early research and development).*

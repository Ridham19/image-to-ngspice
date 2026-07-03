# ⚡ Image-to-NGSPICE (Handwritten Circuit to Simulation)

> 🚧 **WORK IN PROGRESS:** This project is currently under active development. The codebase, pipeline, and features are subject to rapid changes and restructuring. 🚧

## 📖 Overview
This project aims to completely bridge the gap between physical circuit sketches and simulation software. It is an end-to-end Machine Learning pipeline designed to take an image of a handwritten circuit diagram, detect the electronic components, trace the wire routing, and automatically generate a fully executable **NGSPICE netlist**. It now includes both a web-based and desktop-based studio for editing and simulating the recognized circuits.

## ✨ Core Pipeline
1. **Object Detection:** Utilizes a custom-trained YOLO model to identify hand-drawn components (Resistors, Capacitors, Diodes, Voltage Sources, Ground, etc.).
2. **OCR Integration:** Uses EasyOCR for text extraction (component values) near the detected components, with an integrated handwriting autocorrect.
3. **Wire Tracing & Node Extraction:** Leverages `OpenCV` to isolate the wire masks, extract spatial relationships, and determine topological nodes using graph traversal algorithms.
4. **Netlist Generation:** Maps the detected components and their corresponding nodes to standard SPICE syntax, automatically assigning the ground node to `0`.

## 🗂️ Project Structure
The project has evolved into a fully-fledged schematic editor and simulation environment:

* **`WebD/` (WebSpice Studio):** The modern, web-based version of the application.
  * `backend/`: FastAPI server handling AI inference (YOLO + OCR), wire tracing, and NGSPICE simulation execution.
  * `frontend/`: Vanilla JS and HTML/CSS web application for schematic editing, uploading circuits, and viewing simulation plots.
* **`PySpice_studio/` (Desktop Studio):** A desktop-based Python application for editing and simulating circuits.
* **`proper/`:** Contains the original, earlier iteration of the core application.
* **`dataset_new/`, `dataset_clean_v1/`, etc.:** Dataset folders used during research, training, and development of the custom YOLO model.
* **`weights/` / `*.pt`:** PyTorch/YOLO model weights used for component detection.

*(Note: The root directory also contains various deprecated scripts, data generation scripts, and testing scripts from earlier development).*

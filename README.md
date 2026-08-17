# ⚡ Image-to-NGSPICE (Handwritten Circuit to Simulation)

> 🚧 **WORK IN PROGRESS:** This project is currently under active development. The codebase, pipeline, and features are subject to rapid changes and restructuring. 🚧

## 🔗 Demo & Preview
* 🌐 **Interactive Frontend Preview:** [Frontend Live Preview](https://ridham19.github.io/image-to-ngspice/)
* 🎥 **Video Demo:** [Watch on YouTube](https://youtu.be/5eEAm4J13wY)

## 📖 Overview
This project aims to completely bridge the gap between physical circuit sketches and simulation software. It is an end-to-end Machine Learning pipeline designed to take an image of a handwritten circuit diagram, detect the electronic components, trace the wire routing, and automatically generate a fully executable **NGSPICE netlist**. It now includes both a web-based and desktop-based studio for editing and simulating the recognized circuits.

## ✨ Core Pipeline
1. **Object Detection:** Utilizes a custom-trained YOLO model to identify hand-drawn components (Resistors, Capacitors, Diodes, Voltage Sources, Ground, etc.).
2. **OCR Integration:** Uses EasyOCR for text extraction (component values) near the detected components, with an integrated handwriting autocorrect.
3. **Wire Tracing & Node Extraction:** Leverages `OpenCV` to isolate the wire masks, extract spatial relationships, and determine topological nodes using graph traversal algorithms.
4. **Netlist Generation:** Maps the detected components and their corresponding nodes to standard SPICE syntax, automatically assigning the ground node to `0`.

## 🚀 WebSpice Studio (WebD) Features
The modern WebSpice Studio includes advanced editing and verification tools:
* **Interactive Canvas Editor:** Move, rotate, delete, or add components, and draw/route wires in real-time.
* **First-Class Junction Support:** Drawing a wire and snapping it to another wire body automatically inserts a single-pin `junction` component to establish a clear schematic node. Redundant junctions are dynamically cleaned up.
* **Bandwidth-Optimized Simulation Logs:** Displays a bottom console drawer with warnings/errors parsed from `ngspice`'s execution outputs. The raw simulation log is lazy-loaded on-demand only when requested.
* **Collision-Aware Routing:** Custom routing engine with path detouring that routes around physical electronic components while treating auxiliary components (labels, junctions, crossovers) as transparent.
* **Inward Routing Prevention:** Exclude list logic that prevents routed wires from going inwards and cutting through component bodies.

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

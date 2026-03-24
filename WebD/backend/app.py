import os
import shutil
import json
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

# Import your ML Logic (Assuming you copied your 'proper/modules' into 'backend/core')
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.model import ComponentDetector

app = FastAPI(title="Image-to-SPICE API")

# --- CORS SETUP ---
# This allows your HTML/JS frontend to talk to this Python backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, change to your actual web domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- WORKSPACE DIRECTORIES ---
WORKSPACE_DIR = os.path.join(os.path.dirname(__file__), "workspace")
UPLOAD_DIR = os.path.join(WORKSPACE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load the model into memory once when the server starts!
print("🚀 Booting up API Server...")
detector = ComponentDetector(model_name="../weights/best.pt")

@app.get("/")
def read_root():
    return {"status": "Online", "message": "Welcome to the Image-to-SPICE Engine"}

@app.post("/api/detect")
async def detect_circuit(file: UploadFile = File(...)):
    try:
        # 1. Save File
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. Run YOLO + OCR
        json_output_path = os.path.join(WORKSPACE_DIR, "latest_detection.json")
        detected_comps = detector.detect(file_location, output_file=json_output_path)

        # 3. Run OpenCV Wire Tracing
        try:
            from core.processing import preprocess_image, separate_layers
            from core.netlist import trace_nodes
            
            original, gray, binary = preprocess_image(file_location)
            _, wire_mask, _ = separate_layers(gray, binary)
            connections = trace_nodes(wire_mask, detected_comps)
        except Exception as e:
            print(f"⚠️ Wire tracing failed, skipping connections: {e}")
            connections = None

        # 4. Make Data Web-Safe and Return
        safe_components = jsonable_encoder(detected_comps)
        safe_connections = jsonable_encoder(connections)

        return JSONResponse(content={
            "status": "success",
            "components": safe_components,
            "connections": safe_connections
        })

    except Exception as e:
        import traceback
        traceback.print_exc() 
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


if __name__ == "__main__":
    import uvicorn
    # Run the server on port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
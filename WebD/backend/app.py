import os
import shutil
import json
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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
    """
    Receives an image from the frontend, runs YOLO + OCR, and returns the component JSON.
    """
    try:
        # 1. Save the uploaded image temporarily
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"📸 Received image: {file.filename}")

        # 2. Run the ML Pipeline
        # We save a temporary JSON in the workspace, but we will also return the data directly
        json_output_path = os.path.join(WORKSPACE_DIR, "latest_detection.json")
        detected_comps = detector.detect(file_location, output_file=json_output_path)

        # 3. Clean up the uploaded image (optional, to save space)
        # os.remove(file_location)

        # 4. Return the data to the web browser!
        return JSONResponse(content={
            "status": "success",
            "components": detected_comps
        })

    except Exception as e:
        print(f"❌ Error during detection: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    # Run the server on port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
from ultralytics import YOLO

# Load your custom trained model
model = YOLO("best.pt")

# # Export it to ONNX format
# model.export(format="onnx")

metrics = model.val(data="dataset.yaml")

print("✅ Validation complete! Check the 'runs/detect/val' folder.")
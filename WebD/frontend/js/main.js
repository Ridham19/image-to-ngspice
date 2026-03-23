document.addEventListener("DOMContentLoaded", () => {
    const canvas = document.getElementById("circuitCanvas");
    const ctx = canvas.getContext("2d");
    const btnLoadImage = document.getElementById("btnLoadImage");
    const fileInput = document.getElementById("fileInput");
    const statusText = document.getElementById("statusText");

    // Dynamic Canvas Resizing
    function resizeCanvas() {
        canvas.width = canvas.parentElement.clientWidth;
        canvas.height = canvas.parentElement.clientHeight;
        drawGrid();
    }
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    // 1. Draw the background grid (like PySpice Studio)
    function drawGrid() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#444444";
        const gridSize = 20;
        for (let x = 0; x < canvas.width; x += gridSize) {
            for (let y = 0; y < canvas.height; y += gridSize) {
                ctx.fillRect(x, y, 1, 1);
            }
        }
    }

    // 2. Button triggers the hidden file input
    btnLoadImage.addEventListener("click", () => {
        fileInput.click();
    });

    // 3. Handle the File Upload to FastAPI
    fileInput.addEventListener("change", async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        statusText.innerText = "🤖 AI is processing image... Please wait.";
        
        const formData = new FormData();
        formData.append("file", file);

        try {
            // Send to our Python Backend!
            const response = await fetch("http://localhost:8000/api/detect", {
                method: "POST",
                body: formData
            });

            const data = await response.json();
            
            if (data.status === "success") {
                statusText.innerText = `✅ Success! Found ${data.components.length} components.`;
                renderComponents(data.components);
            } else {
                statusText.innerText = "❌ Error: " + data.message;
            }
        } catch (error) {
            statusText.innerText = "❌ Network Error. Is the Python server running?";
            console.error(error);
        }
    });

    // 4. Draw the AI components onto the HTML Canvas
    function renderComponents(components) {
        drawGrid(); // Clear and redraw grid
        
        components.forEach(comp => {
            if (comp.type === 'wire' || comp.type === 'junction' || comp.type === 'text') return;

            // Snap to grid coordinates (assuming center format)
            const gridSize = 20;
            const cx = Math.round(comp.center[0] / gridSize) * gridSize;
            const cy = Math.round(comp.center[1] / gridSize) * gridSize;

            // Draw a placeholder box for the component
            ctx.strokeStyle = "#0078D7";
            ctx.lineWidth = 2;
            ctx.strokeRect(cx - 20, cy - 20, 40, 40);

            // Draw the Component Name & Value (OCR)
            ctx.fillStyle = "#E0E0E0";
            ctx.font = "12px Arial";
            ctx.textAlign = "center";
            ctx.fillText(comp.name, cx, cy - 25);
            
            if (comp.value) {
                ctx.fillStyle = "#FF9800"; // Orange for OCR text
                ctx.fillText(comp.value, cx, cy + 35);
            }
        });
    }
});
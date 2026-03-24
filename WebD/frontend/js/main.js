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
            const response = await fetch("http://127.0.0.1:8000/api/detect", {
                method: "POST",
                body: formData
            });

            const data = await response.json();
            console.log("🤖 AI Response Data:", data); // <--- THIS IS MAGIC. Check your F12 Console!
            
            if (data.status === "success") {
                statusText.innerText = `✅ Success! Found ${data.components.length} components.`;
                // Pass BOTH components and connections to the renderer
                renderComponents(data.components, data.connections); 
            } else {
                statusText.innerText = "❌ Error: " + data.message;
            }
        } catch (error) {
            statusText.innerText = "❌ Network Error.";
            console.error("Fetch Error:", error);
        }
    });

    // 4. Draw the AI components onto the HTML Canvas
    function renderComponents(components, connections) {
    drawGrid(); 
    const gridSize = 20;
    const compMap = {}; // To map AI indices to coordinates

    components.forEach((comp, index) => {
        if (['wire', 'junction', 'text'].includes(comp.type)) return;

        const cx = Math.round(comp.center[0] / gridSize) * gridSize;
        const cy = Math.round(comp.center[1] / gridSize) * gridSize;
        compMap[index] = { x: cx, y: cy, type: comp.type };

        // Draw Component Box
        ctx.strokeStyle = "#0078D7";
        ctx.strokeRect(cx - 20, cy - 20, 40, 40);
        ctx.fillStyle = "#E0E0E0";
        ctx.fillText(comp.name, cx, cy - 25);
    });

    // --- Drawing the Wires ---
    if (connections) {
        ctx.strokeStyle = "#4FC1FF"; // Electric blue wires
        ctx.lineWidth = 2;
        
        connections.forEach((nodes, compIdx) => {
            const startComp = compMap[compIdx];
            if (!startComp) return;

            nodes.forEach(nodeId => {
                // Find other components sharing this node
                connections.forEach((targetNodes, targetIdx) => {
                    if (compIdx === targetIdx) return;
                    if (targetNodes.includes(nodeId)) {
                        const endComp = compMap[targetIdx];
                        if (endComp) {
                            // Simple Manhattan routing (L-shape)
                            ctx.beginPath();
                            ctx.moveTo(startComp.x, startComp.y);
                            ctx.lineTo(startComp.x, endComp.y);
                            ctx.lineTo(endComp.x, endComp.y);
                            ctx.stroke();
                        }
                    }
                });
            });
        });
    }
}
});
document.addEventListener("DOMContentLoaded", () => {
    const canvas = document.getElementById("circuitCanvas");
    const ctx = canvas.getContext("2d");
    const wrapper = document.getElementById("canvasWrapper");
    
    // --- STATE MANAGER ---
    let components = [];
    let wires = [];
    let mode = 'select'; // select, wire, resistor, etc.
    let gridSize = 20;

    // --- CAMERA & VIEWPORT (Pan / Zoom) ---
    let zoom = 1.0;
    let offsetX = 0;
    let offsetY = 0;
    let isPanning = false;
    let panStart = { x: 0, y: 0 };

    // --- INTERACTION STATE ---
    let selectedComp = null;
    let isDragging = false;
    let dragStart = { x: 0, y: 0 };
    let wireStart = null;
    let mousePos = { x: 0, y: 0 }; // World coordinates

    // Resize Canvas to fit screen
    function resizeCanvas() {
        canvas.width = wrapper.clientWidth;
        canvas.height = wrapper.clientHeight;
        render();
    }
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    // --- MATH HELPERS ---
    function screenToWorld(x, y) {
        return {
            x: Math.round(((x - offsetX) / zoom) / gridSize) * gridSize,
            y: Math.round(((y - offsetY) / zoom) / gridSize) * gridSize
        };
    }

    function worldToScreen(x, y) {
        return {
            x: (x * zoom) + offsetX,
            y: (y * zoom) + offsetY
        };
    }

    // --- EVENT LISTENERS ---
    
    // 1. Zooming (Scroll Wheel)
    canvas.addEventListener("wheel", (e) => {
        e.preventDefault();
        const mouseBeforeZoom = screenToWorld(e.offsetX, e.offsetY);
        
        zoom *= e.deltaY > 0 ? 0.9 : 1.1;
        zoom = Math.max(0.2, Math.min(zoom, 5.0)); // Clamp zoom
        
        // Keep mouse pointed at same world coordinate
        offsetX = e.offsetX - (mouseBeforeZoom.x * zoom);
        offsetY = e.offsetY - (mouseBeforeZoom.y * zoom);
        render();
    });

    // 2. Mouse Down (Click, Drag, Pan)
    canvas.addEventListener("mousedown", (e) => {
        const worldPos = screenToWorld(e.offsetX, e.offsetY);

        // Middle Click to Pan
        if (e.button === 1) {
            isPanning = true;
            panStart = { x: e.offsetX, y: e.offsetY };
            canvas.style.cursor = 'grabbing';
            return;
        }

        if (e.button === 0) { // Left Click
            if (mode === 'select') {
                // Check if we clicked a component
                selectedComp = components.find(c => Math.abs(c.x - worldPos.x) <= 20 && Math.abs(c.y - worldPos.y) <= 20);
                if (selectedComp) {
                    isDragging = true;
                    dragStart = { x: worldPos.x, y: worldPos.y };
                }
            } 
            else if (mode === 'wire') {
                if (!wireStart) wireStart = worldPos;
                else {
                    // Draw orthogonal (Manhattan) wire
                    wires.push([{x: wireStart.x, y: wireStart.y}, {x: worldPos.x, y: wireStart.y}]);
                    wires.push([{x: worldPos.x, y: wireStart.y}, {x: worldPos.x, y: worldPos.y}]);
                    wireStart = worldPos; // Chain wires
                }
            }
            else {
                // Place a new component
                components.push({
                    type: mode,
                    x: worldPos.x,
                    y: worldPos.y,
                    name: `${mode.charAt(0).toUpperCase()}?`,
                    value: '1k'
                });
                mode = 'select'; // Revert to select after placing
                updateToolUI();
            }
            render();
        }
    });

    // 3. Mouse Move
    canvas.addEventListener("mousemove", (e) => {
        mousePos = screenToWorld(e.offsetX, e.offsetY);

        if (isPanning) {
            offsetX += e.offsetX - panStart.x;
            offsetY += e.offsetY - panStart.y;
            panStart = { x: e.offsetX, y: e.offsetY };
        } 
        else if (isDragging && selectedComp) {
            const dx = mousePos.x - dragStart.x;
            const dy = mousePos.y - dragStart.y;
            selectedComp.x += dx;
            selectedComp.y += dy;
            dragStart = { x: mousePos.x, y: mousePos.y };
        }
        render(); // Needed to draw temporary wires / ghost components
    });

    // 4. Mouse Up
    canvas.addEventListener("mouseup", (e) => {
        if (e.button === 1) {
            isPanning = false;
            canvas.style.cursor = 'crosshair';
        }
        isDragging = false;
    });

    // 5. Right Click (Cancel Action)
    canvas.addEventListener("contextmenu", (e) => {
        e.preventDefault();
        wireStart = null;
        selectedComp = null;
        mode = 'select';
        updateToolUI();
        render();
    });

    // --- RENDERING ENGINE ---
    function render() {
        // Clear Canvas
        ctx.fillStyle = "#1E1E1E";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw Grid
        ctx.fillStyle = "#444444";
        const step = gridSize * zoom;
        if (step > 5) { // Hide grid if zoomed out too far
            const startX = offsetX % step;
            const startY = offsetY % step;
            for (let x = startX; x < canvas.width; x += step) {
                for (let y = startY; y < canvas.height; y += step) {
                    ctx.fillRect(x, y, 1, 1);
                }
            }
        }

        // Draw Wires
        ctx.strokeStyle = "#4FC1FF";
        ctx.lineWidth = Math.max(1, 2 * zoom);
        wires.forEach(wire => {
            const p1 = worldToScreen(wire[0].x, wire[0].y);
            const p2 = worldToScreen(wire[1].x, wire[1].y);
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();
            
            // Draw nodes at ends
            ctx.fillStyle = "#4FC1FF";
            ctx.beginPath(); ctx.arc(p1.x, p1.y, 3*zoom, 0, Math.PI*2); ctx.fill();
            ctx.beginPath(); ctx.arc(p2.x, p2.y, 3*zoom, 0, Math.PI*2); ctx.fill();
        });

        // Draw Components
        components.forEach(comp => {
            const pos = worldToScreen(comp.x, comp.y);
            const size = 40 * zoom;

            // Highlight if selected
            if (comp === selectedComp) {
                ctx.strokeStyle = "#0078D7";
                ctx.lineWidth = 2;
                ctx.strokeRect(pos.x - size/2 - 5, pos.y - size/2 - 5, size + 10, size + 10);
            }

            // Draw Body
            ctx.strokeStyle = "white";
            ctx.fillStyle = "#2D2D2D";
            ctx.lineWidth = 2 * zoom;
            ctx.fillRect(pos.x - size/2, pos.y - size/2, size, size);
            ctx.strokeRect(pos.x - size/2, pos.y - size/2, size, size);

            // Draw Text
            ctx.fillStyle = "#E0E0E0";
            ctx.font = `${12 * zoom}px Arial`;
            ctx.textAlign = "center";
            ctx.fillText(comp.name, pos.x, pos.y - size/2 - 10);
            if (comp.value) {
                ctx.fillStyle = "#FF9800";
                ctx.fillText(comp.value, pos.x, pos.y + size/2 + 20);
            }
        });

        // Draw Temp Wire (if routing)
        if (mode === 'wire' && wireStart) {
            const p1 = worldToScreen(wireStart.x, wireStart.y);
            const p2 = worldToScreen(mousePos.x, wireStart.y); // Manhattan routing part 1
            const p3 = worldToScreen(mousePos.x, mousePos.y);  // Manhattan routing part 2
            
            ctx.strokeStyle = "rgba(79, 193, 255, 0.5)";
            ctx.setLineDash([5, 5]);
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.lineTo(p3.x, p3.y);
            ctx.stroke();
            ctx.setLineDash([]); // Reset
        }
    }

    // --- TOOLBAR LOGIC ---
    document.querySelectorAll('.btn-tool').forEach(btn => {
        btn.addEventListener('click', (e) => {
            mode = e.target.id.replace('tool-', '');
            wireStart = null;
            updateToolUI();
            render();
        });
    });

    function updateToolUI() {
        document.querySelectorAll('.btn-tool').forEach(btn => btn.classList.remove('active'));
        document.getElementById(`tool-${mode}`).classList.add('active');
    }

    // --- AI IMPORT LOGIC (Re-integrated) ---
    const fileInput = document.getElementById("fileInput");
    document.getElementById("btnLoadImage").addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        document.getElementById("statusText").innerText = "🤖 Processing Image...";
        const formData = new FormData(); formData.append("file", file);

        try {
            const response = await fetch("http://127.0.0.1:8000/api/detect", { method: "POST", body: formData });
            const data = await response.json();
            
            if (data.status === "success") {
                document.getElementById("statusText").innerText = `✅ Loaded ${data.components.length} components.`;
                
                // Load AI Data into our interactive State Engine!
                components = data.components.filter(c => !['wire', 'junction', 'text'].includes(c.type)).map(c => ({
                    type: c.type,
                    name: c.name,
                    value: c.value === "TEXT_FOUND" ? "OCR_WAIT" : c.value,
                    x: Math.round(c.center[0] / gridSize) * gridSize,
                    y: Math.round(c.center[1] / gridSize) * gridSize
                }));

                // Auto-Center Camera
                offsetX = 100; offsetY = 100; zoom = 1.0;
                render();
            }
        } catch (err) { console.error(err); }
    });

    // Initial Draw
    render();
});
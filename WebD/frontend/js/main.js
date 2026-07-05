/**
 * WebSpice Studio — Main Application
 * Complete schematic editor with symbol rendering, properties inspector,
 * and simulation integration.
 */
document.addEventListener("DOMContentLoaded", () => {
    const canvas = document.getElementById("circuitCanvas");
    const ctx = canvas.getContext("2d");
    const wrapper = document.getElementById("canvasWrapper");

    // ═══════════════════════════════════════════
    // COMPONENT DATABASE
    // Mirrors PySpice_studio/library.py DB
    // ═══════════════════════════════════════════
    const COMPONENT_DB = {
        resistor: {
            prefix: 'R', label: 'Resistor',
            params: { value: '1k' },
            spice: '{name} {n1} {n2} {value}',
            pins: [[-40, 0], [40, 0]],   // relative offsets from center
            hitbox: { w: 80, h: 40 }
        },
        capacitor: {
            prefix: 'C', label: 'Capacitor',
            params: { value: '1u', ic: '0' },
            spice: '{name} {n1} {n2} {value} ic={ic}',
            pins: [[-40, 0], [40, 0]],
            hitbox: { w: 80, h: 40 }
        },
        inductor: {
            prefix: 'L', label: 'Inductor',
            params: { value: '1m', ic: '0' },
            spice: '{name} {n1} {n2} {value} ic={ic}',
            pins: [[-40, 0], [40, 0]],
            hitbox: { w: 80, h: 40 }
        },
        diode: {
            prefix: 'D', label: 'Diode',
            params: { model: 'Dx' },
            spice: '{name} {n1} {n2} {model}',
            pins: [[-40, 0], [40, 0]],
            hitbox: { w: 80, h: 40 }
        },
        source: {
            prefix: 'V', label: 'DC Voltage Source',
            params: { dc: '5' },
            spice: '{name} {n1} {n2} DC {dc}',
            pins: [[0, -40], [0, 40]],
            hitbox: { w: 50, h: 80 }
        },
        voltage_source: {
            prefix: 'V', label: 'DC Voltage Source',
            params: { dc: '5' },
            spice: '{name} {n1} {n2} DC {dc}',
            pins: [[0, -40], [0, 40]],
            hitbox: { w: 50, h: 80 }
        },
        current_source: {
            prefix: 'I', label: 'DC Current Source',
            params: { dc: '1m' },
            spice: '{name} {n1} {n2} DC {dc}',
            pins: [[0, -40], [0, 40]],
            hitbox: { w: 50, h: 80 }
        },
        ac_source: {
            prefix: 'V', label: 'AC Source',
            params: { mag: '1', phase: '0' },
            spice: '{name} {n1} {n2} AC {mag} {phase}',
            pins: [[0, -40], [0, 40]],
            hitbox: { w: 50, h: 80 }
        },
        pulse_source: {
            prefix: 'V', label: 'Pulse Source',
            params: { v1: '0', v2: '5', td: '0', tr: '1n', tf: '1n', pw: '10u', per: '20u' },
            spice: '{name} {n1} {n2} PULSE({v1} {v2} {td} {tr} {tf} {pw} {per})',
            pins: [[0, -40], [0, 40]],
            hitbox: { w: 50, h: 80 }
        },
        sine_source: {
            prefix: 'V', label: 'Sine Source',
            params: { vo: '0', va: '5', freq: '1k', td: '0', theta: '0', phase: '0' },
            spice: '{name} {n1} {n2} SINE({vo} {va} {freq} {td} {theta} {phase})',
            pins: [[0, -40], [0, 40]],
            hitbox: { w: 50, h: 80 }
        },
        exp_source: {
            prefix: 'V', label: 'Exp Source',
            params: { v1: '0', v2: '5', td1: '2u', tau1: '2u', td2: '5u', tau2: '5u' },
            spice: '{name} {n1} {n2} EXP({v1} {v2} {td1} {tau1} {td2} {tau2})',
            pins: [[0, -40], [0, 40]],
            hitbox: { w: 50, h: 80 }
        },
        pwl_source: {
            prefix: 'V', label: 'PWL Source',
            params: { pwl_data: '0 0 1m 5' },
            spice: '{name} {n1} {n2} PWL({pwl_data})',
            pins: [[0, -40], [0, 40]],
            hitbox: { w: 50, h: 80 }
        },
        sffm_source: {
            prefix: 'V', label: 'SFFM Source',
            params: { vo: '0', va: '1', fc: '1k', mdi: '5', fs: '200' },
            spice: '{name} {n1} {n2} SFFM({vo} {va} {fc} {mdi} {fs})',
            pins: [[0, -40], [0, 40]],
            hitbox: { w: 50, h: 80 }
        },
        am_source: {
            prefix: 'V', label: 'AM Source',
            params: { va: '5', fc: '1k', mf: '100', ph: '0' },
            spice: '{name} {n1} {n2} AM({va} {fc} {mf} {ph})',
            pins: [[0, -40], [0, 40]],
            hitbox: { w: 50, h: 80 }
        },
        ground: {
            prefix: 'GND', label: 'Ground',
            params: {},
            spice: '',
            pins: [[0, -20]],
            hitbox: { w: 40, h: 30 }
        },
        bjt_npn: {
            prefix: 'Q', label: 'NPN BJT',
            params: { model: 'Tx' },
            spice: '{name} {n2} {n1} {n3} {model}',
            pins: [[-20, 0], [20, -40], [20, 40]],  // Base, Collector, Emitter
            hitbox: { w: 60, h: 80 }
        },
        bjt_pnp: {
            prefix: 'Q', label: 'PNP BJT',
            params: { model: 'Tx_pnp' },
            spice: '{name} {n2} {n1} {n3} {model}',
            pins: [[-20, 0], [20, -40], [20, 40]],
            hitbox: { w: 60, h: 80 }
        },
        bjt: {  // Generic BJT alias (from AI detection)
            prefix: 'Q', label: 'NPN BJT',
            params: { model: 'Tx' },
            spice: '{name} {n2} {n1} {n3} {model}',
            pins: [[-20, 0], [20, -40], [20, 40]],
            hitbox: { w: 60, h: 80 }
        },
        transistor: {  // YOLO model class label
            prefix: 'Q', label: 'NPN BJT',
            params: { model: 'Tx' },
            spice: '{name} {n2} {n1} {n3} {model}',
            pins: [[-20, 0], [20, -40], [20, 40]],
            hitbox: { w: 60, h: 80 }
        },
        // ═══════════════════════════════════════════
        // NEW — epoch_40.pt model classes
        // ═══════════════════════════════════════════
        vss: {
            prefix: 'V', label: 'VSS Supply',
            params: { dc: '-5' },
            spice: '{name} {n1} {n2} DC {dc}',
            pins: [[0, -40], [0, 40]],
            hitbox: { w: 50, h: 80 }
        },
        capacitor_polarized: {
            prefix: 'C', label: 'Polarized Capacitor',
            params: { value: '10u', ic: '0' },
            spice: '{name} {n1} {n2} {value} ic={ic}',
            pins: [[-40, 0], [40, 0]],
            hitbox: { w: 80, h: 40 }
        },
        resistor_photo: {
            prefix: 'R', label: 'Photoresistor (LDR)',
            params: { value: '10k' },
            spice: '{name} {n1} {n2} {value}',
            pins: [[-40, 0], [40, 0]],
            hitbox: { w: 80, h: 40 }
        },
        diode_led: {
            prefix: 'D', label: 'LED',
            params: { model: 'Dx' },
            spice: '{name} {n1} {n2} {model}',
            pins: [[-40, 0], [40, 0]],
            hitbox: { w: 80, h: 40 }
        },
        diode_zener: {
            prefix: 'D', label: 'Zener Diode',
            params: { model: 'Dx' },
            spice: '{name} {n1} {n2} {model}',
            pins: [[-40, 0], [40, 0]],
            hitbox: { w: 80, h: 40 }
        },
        mosfet: {
            prefix: 'M', label: 'N-MOSFET',
            params: { model: 'Mx' },
            spice: '{name} {n2} {n1} {n3} {n3} {model}',
            pins: [[-20, 0], [20, -40], [20, 40]],  // Gate, Drain, Source
            hitbox: { w: 60, h: 80 }
        },
        phototransistor: {
            prefix: 'Q', label: 'Phototransistor',
            params: { model: 'Tx' },
            spice: '{name} {n2} {n1} {n3} {model}',
            pins: [[-20, 0], [20, -40], [20, 40]],
            hitbox: { w: 60, h: 80 }
        },
        opamp: {
            prefix: 'U', label: 'Op-Amp',
            params: {},
            spice: '',
            pins: [[-30, -20], [-30, 20], [30, 0]],  // V+, V−, Out
            hitbox: { w: 70, h: 60 }
        },
        ic: {
            prefix: 'U', label: 'IC',
            params: {},
            spice: '',
            pins: [[-40, 0], [40, 0]],
            hitbox: { w: 80, h: 50 }
        },
        transformer: {
            prefix: 'T', label: 'Transformer',
            params: {},
            spice: '',
            pins: [[-40, -20], [-40, 20], [40, -20], [40, 20]],
            hitbox: { w: 80, h: 60 }
        },
        junction: {
            prefix: 'J', label: 'Junction',
            params: {},
            spice: '',
            pins: [[0, 0]],
            hitbox: { w: 10, h: 10 }
        },
        crossover: {
            prefix: 'X', label: 'Crossover',
            params: {},
            spice: '',
            pins: [[0, 0]],
            hitbox: { w: 20, h: 20 }
        },
        terminal: {
            prefix: 'P', label: 'Terminal',
            params: {},
            spice: '',
            pins: [[0, 0]],
            hitbox: { w: 20, h: 20 }
        },
        label: {
            prefix: 'LBL', label: 'Node Label',
            params: { name: 'Vout' },
            spice: '',
            pins: [[0, 0]],
            hitbox: { w: 40, h: 20 }
        }
    };

    // ═══════════════════════════════════════════
    // STATE MANAGER
    // ═══════════════════════════════════════════
    let components = [];
    let wires = [];
    let mode = 'select';
    const gridSize = 20;
    let nameCounts = {};  // Per-prefix auto-naming counters

    // Simulation config state (mirrors PySpice Studio's sim_data)
    let simConfig = {
        mode: 'op',
        params: {},
        plots: {},    // window_id → [signal_names]
        colors: {},   // color_index → color_name
    };
    let plotSignals = [];  // [{signal, color, window}]
    // Available nodes/sources/sweepables (populated after solving)
    let availableNodes = [];
    let availableSources = [];
    let availableSweepables = [];

    // Camera & Viewport
    let zoom = 1.0;
    let offsetX = 0;
    let offsetY = 0;
    let isPanning = false;
    let panStart = { x: 0, y: 0 };

    // Interaction State
    let selectedComp = null;
    let isDragging = false;
    let dragStart = { x: 0, y: 0 };
    let wireStart = null;
    let mousePos = { x: 0, y: 0 };
    let attachedWireEndpoints = [];

    // ═══════════════════════════════════════════
    // CANVAS RESIZE
    // ═══════════════════════════════════════════
    function resizeCanvas() {
        canvas.width = wrapper.clientWidth;
        canvas.height = wrapper.clientHeight;
        render();
    }
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    // ═══════════════════════════════════════════
    // MATH HELPERS
    // ═══════════════════════════════════════════
    function snap(val) { return Math.round(val / gridSize) * gridSize; }

    /** Convert screen coords to world coords WITHOUT snapping */
    function screenToWorldRaw(x, y) {
        return {
            x: (x - offsetX) / zoom,
            y: (y - offsetY) / zoom
        };
    }

    /** Convert screen coords to world coords WITH grid snapping */
    function screenToWorld(x, y) {
        const raw = screenToWorldRaw(x, y);
        return { x: snap(raw.x), y: snap(raw.y) };
    }

    function worldToScreen(x, y) {
        return {
            x: (x * zoom) + offsetX,
            y: (y * zoom) + offsetY
        };
    }

    // Compute absolute pin positions for a component
    function getCompPins(comp) {
        const db = COMPONENT_DB[comp.type];
        if (!db) return [{ x: comp.x, y: comp.y }];
        return db.pins.map(([dx, dy]) => ({
            x: comp.x + dx,
            y: comp.y + dy
        }));
    }

    /**
     * PIN SNAP RADIUS — if the cursor is within this many world-units
     * of a component pin, the wire endpoint locks onto that pin.
     */
    const PIN_SNAP_RADIUS = 15;

    /** Find the nearest component pin to a raw world position.
     *  Returns {x, y} of the pin if within PIN_SNAP_RADIUS, else null. */
    function findNearestPin(rawWorldX, rawWorldY) {
        let bestDist = PIN_SNAP_RADIUS;
        let bestPin = null;

        for (const comp of components) {
            const pins = getCompPins(comp);
            for (const pin of pins) {
                const dx = rawWorldX - pin.x;
                const dy = rawWorldY - pin.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestPin = { x: pin.x, y: pin.y };
                }
            }
        }

        // Also snap to existing wire endpoints
        for (const wire of wires) {
            for (const pt of wire) {
                const dx = rawWorldX - pt.x;
                const dy = rawWorldY - pt.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestPin = { x: pt.x, y: pt.y };
                }
            }
        }

        return bestPin;
    }

    /** Find the nearest point along any wire segment (T-junction snapping) */
    function findNearestWireSegment(rawWorldX, rawWorldY) {
        let bestDist = PIN_SNAP_RADIUS;
        let bestPt = null;
        for (const wire of wires) {
            for (let i = 0; i < wire.length - 1; i++) {
                const p1 = wire[i];
                const p2 = wire[i+1];
                const l2 = (p1.x - p2.x)**2 + (p1.y - p2.y)**2;
                if (l2 === 0) continue;
                
                // Point to line segment projection
                let t = ((rawWorldX - p1.x) * (p2.x - p1.x) + (rawWorldY - p1.y) * (p2.y - p1.y)) / l2;
                t = Math.max(0, Math.min(1, t));
                const projX = p1.x + t * (p2.x - p1.x);
                const projY = p1.y + t * (p2.y - p1.y);
                
                const dist = Math.sqrt((rawWorldX - projX)**2 + (rawWorldY - projY)**2);
                if (dist < bestDist) {
                    bestDist = dist;
                    // Snap the intersection point strictly to the grid to ensure netlist continuity
                    bestPt = { x: snap(projX), y: snap(projY) };
                }
            }
        }
        return bestPt;
    }

    /** Smart snap: prefer nearest pin, then nearest wire body (T-junction), fall back to grid. */
    function smartSnap(screenX, screenY) {
        const raw = screenToWorldRaw(screenX, screenY);
        const pinHit = findNearestPin(raw.x, raw.y);
        if (pinHit) return pinHit;
        
        const wireHit = findNearestWireSegment(raw.x, raw.y);
        if (wireHit) return wireHit;
        
        return { x: snap(raw.x), y: snap(raw.y) };
    }

    // Track whether we're currently snapped to a pin (for visual feedback)
    let hoveredPin = null;

    // Generate the next auto-name for a component prefix
    function nextName(prefix) {
        nameCounts[prefix] = (nameCounts[prefix] || 0) + 1;
        return `${prefix}${nameCounts[prefix]}`;
    }

    // Create a new component object from its type and position
    function createComponent(type, worldX, worldY) {
        const db = COMPONENT_DB[type];
        if (!db) {
            return {
                type, x: worldX, y: worldY,
                name: type.charAt(0).toUpperCase() + '?',
                value: '1k', params: { value: '1k' }
            };
        }
        const name = nextName(db.prefix);
        const params = Object.assign({}, db.params);
        return {
            type, x: worldX, y: worldY,
            name,
            value: params.value || params.dc || params.mag || '',
            params
        };
    }

    // ═══════════════════════════════════════════
    // EVENT LISTENERS
    // ═══════════════════════════════════════════

    // 1. Zooming (Scroll Wheel)
    canvas.addEventListener("wheel", (e) => {
        e.preventDefault();
        const mx = (e.offsetX - offsetX) / zoom;
        const my = (e.offsetY - offsetY) / zoom;

        zoom *= e.deltaY > 0 ? 0.9 : 1.1;
        zoom = Math.max(0.2, Math.min(zoom, 5.0));

        offsetX = e.offsetX - (mx * zoom);
        offsetY = e.offsetY - (my * zoom);
        render();
    });

    // 2. Mouse Down
    canvas.addEventListener("mousedown", (e) => {
        // Middle Click to Pan
        if (e.button === 1) {
            isPanning = true;
            panStart = { x: e.offsetX, y: e.offsetY };
            canvas.style.cursor = 'grabbing';
            return;
        }

        if (e.button === 0) {
            if (mode === 'select') {
                const worldPos = screenToWorld(e.offsetX, e.offsetY);
                const hit = hitTest(worldPos);
                if (hit) {
                    selectedComp = hit;
                    isDragging = true;
                    dragStart = { x: worldPos.x, y: worldPos.y };
                    
                    // Find attached wire endpoints for rubber-banding
                    const pins = getCompPins(selectedComp);
                    attachedWireEndpoints = [];
                    wires.forEach(wire => {
                        if (wire.length < 2) return;
                        [wire[0], wire[wire.length - 1]].forEach(pt => {
                            const pinIdx = pins.findIndex(pin => Math.abs(pin.x - pt.x) < 1 && Math.abs(pin.y - pt.y) < 1);
                            if (pinIdx !== -1) {
                                attachedWireEndpoints.push({ pt, pinIdx });
                            }
                        });
                    });

                    updatePropertiesPanel();
                } else {
                    selectedComp = null;
                    updatePropertiesPanel();
                }
            }
            else if (mode === 'wire') {
                // Use smart snapping for wire endpoints (pin > grid)
                const snapped = smartSnap(e.offsetX, e.offsetY);
                const raw = screenToWorldRaw(e.offsetX, e.offsetY);
                const landedOnPin = findNearestPin(raw.x, raw.y);

                if (!wireStart) {
                    wireStart = snapped;
                } else {
                    // Manhattan wire routing
                    if (wireStart.x !== snapped.x || wireStart.y !== snapped.y) {
                        wires.push([
                            { x: wireStart.x, y: wireStart.y },
                            { x: snapped.x, y: wireStart.y }
                        ]);
                        if (wireStart.y !== snapped.y) {
                            wires.push([
                                { x: snapped.x, y: wireStart.y },
                                { x: snapped.x, y: snapped.y }
                            ]);
                        }
                    }
                    // If we landed on a pin, auto-terminate the wire
                    if (landedOnPin) {
                        wireStart = null;
                        hoveredPin = null;
                    } else {
                        wireStart = snapped;
                    }
                }
            }
            else {
                // Place a new component (always grid-snap)
                const worldPos = screenToWorld(e.offsetX, e.offsetY);
                const comp = createComponent(mode, worldPos.x, worldPos.y);
                components.push(comp);
                selectedComp = comp;
                mode = 'select';
                updateToolUI();
                updatePropertiesPanel();
            }
            render();
        }
    });

    // 3. Mouse Move
    canvas.addEventListener("mousemove", (e) => {
        if (isPanning) {
            offsetX += e.offsetX - panStart.x;
            offsetY += e.offsetY - panStart.y;
            panStart = { x: e.offsetX, y: e.offsetY };
            mousePos = screenToWorld(e.offsetX, e.offsetY);
            hoveredPin = null;
        }
        else if (isDragging && selectedComp) {
            mousePos = screenToWorld(e.offsetX, e.offsetY);
            const dx = mousePos.x - dragStart.x;
            const dy = mousePos.y - dragStart.y;
            selectedComp.x += dx;
            selectedComp.y += dy;
            
            // Move attached wires
            attachedWireEndpoints.forEach(att => {
                att.pt.x += dx;
                att.pt.y += dy;
            });
            
            dragStart = { x: mousePos.x, y: mousePos.y };
            hoveredPin = null;
        }
        else if (mode === 'wire') {
            // In wire mode, use smart snap so the ghost preview
            // locks to nearby pins
            const snapped = smartSnap(e.offsetX, e.offsetY);
            mousePos = snapped;

            // Track if we're hovering over a pin (for visual feedback)
            const raw = screenToWorldRaw(e.offsetX, e.offsetY);
            hoveredPin = findNearestPin(raw.x, raw.y);
        }
        else {
            mousePos = screenToWorld(e.offsetX, e.offsetY);
            hoveredPin = null;
        }
        render();
    });

    // 4. Mouse Up
    canvas.addEventListener("mouseup", (e) => {
        if (e.button === 1) {
            isPanning = false;
            canvas.style.cursor = 'crosshair';
        }
        if (isDragging && selectedComp) {
            // Snap component back to grid on drop
            selectedComp.x = snap(selectedComp.x);
            selectedComp.y = snap(selectedComp.y);
            
            // Snap attached wire endpoints precisely to the grid-aligned pins
            const snappedPins = getCompPins(selectedComp);
            attachedWireEndpoints.forEach(att => {
                const pin = snappedPins[att.pinIdx];
                att.pt.x = pin.x;
                att.pt.y = pin.y;
            });
            
            // Auto-orthogonalize any wires that became diagonal during drag
            const newWires = [];
            wires.forEach(wire => {
                if (wire.length === 2) {
                    if (wire[0].x !== wire[1].x && wire[0].y !== wire[1].y) {
                        // Split into two orthogonal segments
                        newWires.push([ {x: wire[0].x, y: wire[0].y}, {x: wire[1].x, y: wire[0].y} ]);
                        newWires.push([ {x: wire[1].x, y: wire[0].y}, {x: wire[1].x, y: wire[1].y} ]);
                    } else {
                        newWires.push(wire);
                    }
                } else {
                    newWires.push(wire);
                }
            });
            wires = newWires;
            attachedWireEndpoints = [];
            
            updatePropertiesPanel();
            render();
        }
        isDragging = false;
    });

    // 5. Right Click (Cancel)
    canvas.addEventListener("contextmenu", (e) => {
        e.preventDefault();
        wireStart = null;
        selectedComp = null;
        mode = 'select';
        updateToolUI();
        updatePropertiesPanel();
        render();
    });

    // Keyboard shortcut: Delete selected component
    document.addEventListener("keydown", (e) => {
        if (e.key === 'Delete' && selectedComp) {
            const idx = components.indexOf(selectedComp);
            if (idx !== -1) components.splice(idx, 1);
            selectedComp = null;
            updatePropertiesPanel();
            render();
        }
        if (e.key === 'Escape') {
            wireStart = null;
            selectedComp = null;
            mode = 'select';
            updateToolUI();
            updatePropertiesPanel();
            render();
        }
    });

    // ═══════════════════════════════════════════
    // HIT TESTING (Per-type bounding boxes)
    // ═══════════════════════════════════════════
    function hitTest(worldPos) {
        // Iterate in reverse so topmost (last drawn) is hit first
        for (let i = components.length - 1; i >= 0; i--) {
            const c = components[i];
            const db = COMPONENT_DB[c.type];
            const hb = db ? db.hitbox : { w: 40, h: 40 };
            const hw = hb.w / 2;
            const hh = hb.h / 2;
            if (worldPos.x >= c.x - hw && worldPos.x <= c.x + hw &&
                worldPos.y >= c.y - hh && worldPos.y <= c.y + hh) {
                return c;
            }
        }
        return null;
    }

    // ═══════════════════════════════════════════
    // SYMBOL RENDER ENGINE
    // ═══════════════════════════════════════════
    const SYMBOL_RENDERERS = {
        resistor: drawResistor,
        capacitor: drawCapacitor,
        inductor: drawInductor,
        diode: drawDiode,
        source: drawDCSource,
        voltage_source: drawDCSource,
        current_source: drawCurrentSource,
        ac_source: drawACSource,
        pulse_source: drawPulseSource,
        sine_source: drawSineSource,
        exp_source: drawExpSource,
        pwl_source: drawPWLSource,
        sffm_source: drawSFFMSource,
        am_source: drawAMSource,
        ground: drawGround,
        bjt_npn: drawBJT_NPN,
        bjt_pnp: drawBJT_PNP,
        bjt: drawBJT_NPN,
        transistor: drawBJT_NPN,
        label: drawLabel,
        // ── New epoch_40.pt renderers ──
        vss: drawVSS,
        capacitor_polarized: drawCapacitorPolarized,
        resistor_photo: drawResistorPhoto,
        diode_led: drawDiodeLED,
        diode_zener: drawDiodeZener,
        mosfet: drawMOSFET,
        phototransistor: drawPhototransistor,
        opamp: drawOpAmp,
        ic: drawIC,
        transformer: drawTransformer,
        junction: drawJunction,
        crossover: drawCrossover,
        terminal: drawTerminal
    };

    function drawLabel(ctx, sx, sy, z, comp) {
        // A label is just a small pin marker, the text is drawn elsewhere
        ctx.beginPath();
        ctx.arc(sx, sy, 3 * z, 0, 2 * Math.PI);
        ctx.fill();
    }

    function drawResistor(ctx, sx, sy, z) {
        // Horizontal leads + zig-zag body
        const leadL = 12 * z;
        const bodyW = 56 * z;
        const amp = 8 * z;
        const peaks = 6;
        const segW = bodyW / peaks;

        ctx.beginPath();
        // Left lead
        ctx.moveTo(sx - 40 * z, sy);
        ctx.lineTo(sx - 28 * z, sy);
        // Zig-zag
        let x = sx - 28 * z;
        for (let i = 0; i < peaks; i++) {
            const dir = (i % 2 === 0) ? -1 : 1;
            ctx.lineTo(x + segW / 2, sy + amp * dir);
            ctx.lineTo(x + segW, sy);
            x += segW;
        }
        // Right lead
        ctx.lineTo(sx + 40 * z, sy);
        ctx.stroke();

        // Pin dots
        drawPinDot(ctx, sx - 40 * z, sy, z);
        drawPinDot(ctx, sx + 40 * z, sy, z);
    }

    function drawCapacitor(ctx, sx, sy, z) {
        const gap = 4 * z;
        const plateH = 16 * z;

        ctx.beginPath();
        // Left lead
        ctx.moveTo(sx - 40 * z, sy);
        ctx.lineTo(sx - gap, sy);
        ctx.stroke();

        // Left plate
        ctx.beginPath();
        ctx.moveTo(sx - gap, sy - plateH);
        ctx.lineTo(sx - gap, sy + plateH);
        ctx.stroke();

        // Right plate
        ctx.beginPath();
        ctx.moveTo(sx + gap, sy - plateH);
        ctx.lineTo(sx + gap, sy + plateH);
        ctx.stroke();

        // Right lead
        ctx.beginPath();
        ctx.moveTo(sx + gap, sy);
        ctx.lineTo(sx + 40 * z, sy);
        ctx.stroke();

        drawPinDot(ctx, sx - 40 * z, sy, z);
        drawPinDot(ctx, sx + 40 * z, sy, z);
    }

    function drawInductor(ctx, sx, sy, z) {
        const humps = 4;
        const humpW = 12 * z;
        const totalW = humps * humpW;
        const startX = sx - totalW / 2;

        // Left lead
        ctx.beginPath();
        ctx.moveTo(sx - 40 * z, sy);
        ctx.lineTo(startX, sy);
        ctx.stroke();

        // Arcs (humps)
        for (let i = 0; i < humps; i++) {
            const cx = startX + i * humpW + humpW / 2;
            ctx.beginPath();
            ctx.arc(cx, sy, humpW / 2, Math.PI, 0, false);
            ctx.stroke();
        }

        // Right lead
        ctx.beginPath();
        ctx.moveTo(startX + totalW, sy);
        ctx.lineTo(sx + 40 * z, sy);
        ctx.stroke();

        drawPinDot(ctx, sx - 40 * z, sy, z);
        drawPinDot(ctx, sx + 40 * z, sy, z);
    }

    function drawDiode(ctx, sx, sy, z) {
        const triW = 14 * z;
        const triH = 10 * z;

        // Left lead
        ctx.beginPath();
        ctx.moveTo(sx - 40 * z, sy);
        ctx.lineTo(sx - triW, sy);
        ctx.stroke();

        // Triangle (anode)
        ctx.beginPath();
        ctx.moveTo(sx - triW, sy - triH);
        ctx.lineTo(sx - triW, sy + triH);
        ctx.lineTo(sx + triW, sy);
        ctx.closePath();
        ctx.stroke();

        // Cathode bar
        ctx.beginPath();
        ctx.moveTo(sx + triW, sy - triH);
        ctx.lineTo(sx + triW, sy + triH);
        ctx.stroke();

        // Right lead
        ctx.beginPath();
        ctx.moveTo(sx + triW, sy);
        ctx.lineTo(sx + 40 * z, sy);
        ctx.stroke();

        drawPinDot(ctx, sx - 40 * z, sy, z);
        drawPinDot(ctx, sx + 40 * z, sy, z);
    }

    function drawDCSource(ctx, sx, sy, z) {
        const r = 18 * z;

        // Top lead (pin at [0,-40])
        ctx.beginPath();
        ctx.moveTo(sx, sy - 40 * z);
        ctx.lineTo(sx, sy - r);
        ctx.stroke();

        // Circle
        ctx.beginPath();
        ctx.arc(sx, sy, r, 0, Math.PI * 2);
        ctx.stroke();

        // + sign (top half)
        const signOff = 7 * z;
        ctx.beginPath();
        ctx.moveTo(sx, sy - signOff - 4 * z);
        ctx.lineTo(sx, sy - signOff + 4 * z);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(sx - 4 * z, sy - signOff);
        ctx.lineTo(sx + 4 * z, sy - signOff);
        ctx.stroke();

        // - sign (bottom half)
        ctx.beginPath();
        ctx.moveTo(sx - 4 * z, sy + signOff);
        ctx.lineTo(sx + 4 * z, sy + signOff);
        ctx.stroke();

        // Bottom lead (pin at [0,40])
        ctx.beginPath();
        ctx.moveTo(sx, sy + r);
        ctx.lineTo(sx, sy + 40 * z);
        ctx.stroke();

        drawPinDot(ctx, sx, sy - 40 * z, z);
        drawPinDot(ctx, sx, sy + 40 * z, z);
    }

    function drawCurrentSource(ctx, sx, sy, z) {
        const r = 18 * z;

        // Top lead
        ctx.beginPath();
        ctx.moveTo(sx, sy - 40 * z);
        ctx.lineTo(sx, sy - r);
        ctx.stroke();

        // Circle
        ctx.beginPath();
        ctx.arc(sx, sy, r, 0, Math.PI * 2);
        ctx.stroke();

        // Arrow inside (pointing up)
        ctx.beginPath();
        ctx.moveTo(sx, sy + 10 * z);
        ctx.lineTo(sx, sy - 10 * z);
        ctx.stroke();
        // Arrowhead
        ctx.beginPath();
        ctx.moveTo(sx - 4 * z, sy - 5 * z);
        ctx.lineTo(sx, sy - 10 * z);
        ctx.lineTo(sx + 4 * z, sy - 5 * z);
        ctx.stroke();

        // Bottom lead
        ctx.beginPath();
        ctx.moveTo(sx, sy + r);
        ctx.lineTo(sx, sy + 40 * z);
        ctx.stroke();

        drawPinDot(ctx, sx, sy - 40 * z, z);
        drawPinDot(ctx, sx, sy + 40 * z, z);
    }

    function drawACSource(ctx, sx, sy, z) {
        const r = 18 * z;

        // Top lead
        ctx.beginPath();
        ctx.moveTo(sx, sy - 40 * z);
        ctx.lineTo(sx, sy - r);
        ctx.stroke();

        // Circle
        ctx.beginPath();
        ctx.arc(sx, sy, r, 0, Math.PI * 2);
        ctx.stroke();

        // AC Sine wave inside with AC text
        ctx.beginPath();
        const waveW = 10 * z;
        const waveH = 6 * z;
        ctx.moveTo(sx - waveW, sy);
        ctx.quadraticCurveTo(sx - waveW / 2, sy - waveH * 2, sx, sy);
        ctx.quadraticCurveTo(sx + waveW / 2, sy + waveH * 2, sx + waveW, sy);
        ctx.stroke();

        // Bottom lead
        ctx.beginPath();
        ctx.moveTo(sx, sy + r);
        ctx.lineTo(sx, sy + 40 * z);
        ctx.stroke();

        drawPinDot(ctx, sx, sy - 40 * z, z);
        drawPinDot(ctx, sx, sy + 40 * z, z);
    }

    function drawSineSource(ctx, sx, sy, z) {
        const r = 18 * z;
        ctx.beginPath();
        ctx.moveTo(sx, sy - 40 * z);
        ctx.lineTo(sx, sy - r);
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(sx, sy, r, 0, Math.PI * 2);
        ctx.stroke();

        // Stylized Sine wave
        ctx.beginPath();
        const waveW = 11 * z;
        const waveH = 7 * z;
        ctx.moveTo(sx - waveW, sy);
        ctx.quadraticCurveTo(sx - waveW / 2, sy - waveH * 2, sx, sy);
        ctx.quadraticCurveTo(sx + waveW / 2, sy + waveH * 2, sx + waveW, sy);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(sx, sy + r);
        ctx.lineTo(sx, sy + 40 * z);
        ctx.stroke();

        drawPinDot(ctx, sx, sy - 40 * z, z);
        drawPinDot(ctx, sx, sy + 40 * z, z);
    }

    function drawPulseSource(ctx, sx, sy, z) {
        const r = 18 * z;
        ctx.beginPath();
        ctx.moveTo(sx, sy - 40 * z);
        ctx.lineTo(sx, sy - r);
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(sx, sy, r, 0, Math.PI * 2);
        ctx.stroke();

        // Square pulse shape inside
        ctx.beginPath();
        ctx.moveTo(sx - 10 * z, sy + 5 * z);
        ctx.lineTo(sx - 10 * z, sy - 5 * z);
        ctx.lineTo(sx, sy - 5 * z);
        ctx.lineTo(sx, sy + 5 * z);
        ctx.lineTo(sx + 10 * z, sy + 5 * z);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(sx, sy + r);
        ctx.lineTo(sx, sy + 40 * z);
        ctx.stroke();

        drawPinDot(ctx, sx, sy - 40 * z, z);
        drawPinDot(ctx, sx, sy + 40 * z, z);
    }

    function drawPWLSource(ctx, sx, sy, z) {
        const r = 18 * z;
        ctx.beginPath();
        ctx.moveTo(sx, sy - 40 * z);
        ctx.lineTo(sx, sy - r);
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(sx, sy, r, 0, Math.PI * 2);
        ctx.stroke();

        // Piecewise linear shape (rising ramp/stairs)
        ctx.beginPath();
        ctx.moveTo(sx - 10 * z, sy + 7 * z);
        ctx.lineTo(sx - 4 * z, sy + 7 * z);
        ctx.lineTo(sx + 2 * z, sy - 5 * z);
        ctx.lineTo(sx + 10 * z, sy - 5 * z);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(sx, sy + r);
        ctx.lineTo(sx, sy + 40 * z);
        ctx.stroke();

        drawPinDot(ctx, sx, sy - 40 * z, z);
        drawPinDot(ctx, sx, sy + 40 * z, z);
    }

    function drawExpSource(ctx, sx, sy, z) {
        const r = 18 * z;
        ctx.beginPath();
        ctx.moveTo(sx, sy - 40 * z);
        ctx.lineTo(sx, sy - r);
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(sx, sy, r, 0, Math.PI * 2);
        ctx.stroke();

        // Exponential curve rising shape
        ctx.beginPath();
        ctx.moveTo(sx - 10 * z, sy + 6 * z);
        ctx.quadraticCurveTo(sx + 2 * z, sy + 6 * z, sx + 10 * z, sy - 8 * z);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(sx, sy + r);
        ctx.lineTo(sx, sy + 40 * z);
        ctx.stroke();

        drawPinDot(ctx, sx, sy - 40 * z, z);
        drawPinDot(ctx, sx, sy + 40 * z, z);
    }

    function drawSFFMSource(ctx, sx, sy, z) {
        const r = 18 * z;
        ctx.beginPath();
        ctx.moveTo(sx, sy - 40 * z);
        ctx.lineTo(sx, sy - r);
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(sx, sy, r, 0, Math.PI * 2);
        ctx.stroke();

        // SFFM Label inside
        ctx.fillStyle = "#E0E0E0";
        ctx.font = `bold ${10 * z}px 'Segoe UI', Arial`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("FM", sx, sy);

        ctx.beginPath();
        ctx.moveTo(sx, sy + r);
        ctx.lineTo(sx, sy + 40 * z);
        ctx.stroke();

        drawPinDot(ctx, sx, sy - 40 * z, z);
        drawPinDot(ctx, sx, sy + 40 * z, z);
    }

    function drawAMSource(ctx, sx, sy, z) {
        const r = 18 * z;
        ctx.beginPath();
        ctx.moveTo(sx, sy - 40 * z);
        ctx.lineTo(sx, sy - r);
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(sx, sy, r, 0, Math.PI * 2);
        ctx.stroke();

        // AM Label inside
        ctx.fillStyle = "#E0E0E0";
        ctx.font = `bold ${10 * z}px 'Segoe UI', Arial`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("AM", sx, sy);

        ctx.beginPath();
        ctx.moveTo(sx, sy + r);
        ctx.lineTo(sx, sy + 40 * z);
        ctx.stroke();

        drawPinDot(ctx, sx, sy - 40 * z, z);
        drawPinDot(ctx, sx, sy + 40 * z, z);
    }

    function drawGround(ctx, sx, sy, z) {
        // Lead up from pin at [0,-20]
        ctx.beginPath();
        ctx.moveTo(sx, sy - 20 * z);
        ctx.lineTo(sx, sy);
        ctx.stroke();

        // Three horizontal lines, decreasing width
        const widths = [18, 12, 6];
        for (let i = 0; i < 3; i++) {
            const w = widths[i] * z;
            const y = sy + i * 5 * z;
            ctx.beginPath();
            ctx.moveTo(sx - w, y);
            ctx.lineTo(sx + w, y);
            ctx.stroke();
        }

        drawPinDot(ctx, sx, sy - 20 * z, z);
    }

    function drawBJT_NPN(ctx, sx, sy, z) {
        // Base lead (pin at [-20, 0])
        ctx.beginPath();
        ctx.moveTo(sx - 20 * z, sy);
        ctx.lineTo(sx - 4 * z, sy);
        ctx.stroke();

        // Base vertical bar
        ctx.lineWidth = Math.max(1, 3 * z);
        ctx.beginPath();
        ctx.moveTo(sx - 4 * z, sy - 14 * z);
        ctx.lineTo(sx - 4 * z, sy + 14 * z);
        ctx.stroke();
        ctx.lineWidth = Math.max(1, 2 * z);

        // Collector line (from bar center-top to pin at [20, -40])
        ctx.beginPath();
        ctx.moveTo(sx - 4 * z, sy - 8 * z);
        ctx.lineTo(sx + 14 * z, sy - 24 * z);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(sx + 14 * z, sy - 24 * z);
        ctx.lineTo(sx + 20 * z, sy - 40 * z);
        ctx.stroke();

        // Emitter line (from bar center-bottom to pin at [20, 40])
        ctx.beginPath();
        ctx.moveTo(sx - 4 * z, sy + 8 * z);
        ctx.lineTo(sx + 14 * z, sy + 24 * z);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(sx + 14 * z, sy + 24 * z);
        ctx.lineTo(sx + 20 * z, sy + 40 * z);
        ctx.stroke();

        // Emitter arrow (pointing outward)
        drawArrow(ctx, sx - 4 * z, sy + 8 * z, sx + 14 * z, sy + 24 * z, z);

        // Pin dots
        drawPinDot(ctx, sx - 20 * z, sy, z);
        drawPinDot(ctx, sx + 20 * z, sy - 40 * z, z);
        drawPinDot(ctx, sx + 20 * z, sy + 40 * z, z);
    }

    function drawBJT_PNP(ctx, sx, sy, z) {
        // Base lead
        ctx.beginPath();
        ctx.moveTo(sx - 20 * z, sy);
        ctx.lineTo(sx - 4 * z, sy);
        ctx.stroke();

        // Base vertical bar
        ctx.lineWidth = Math.max(1, 3 * z);
        ctx.beginPath();
        ctx.moveTo(sx - 4 * z, sy - 14 * z);
        ctx.lineTo(sx - 4 * z, sy + 14 * z);
        ctx.stroke();
        ctx.lineWidth = Math.max(1, 2 * z);

        // Collector
        ctx.beginPath();
        ctx.moveTo(sx - 4 * z, sy - 8 * z);
        ctx.lineTo(sx + 14 * z, sy - 24 * z);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(sx + 14 * z, sy - 24 * z);
        ctx.lineTo(sx + 20 * z, sy - 40 * z);
        ctx.stroke();

        // Emitter
        ctx.beginPath();
        ctx.moveTo(sx - 4 * z, sy + 8 * z);
        ctx.lineTo(sx + 14 * z, sy + 24 * z);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(sx + 14 * z, sy + 24 * z);
        ctx.lineTo(sx + 20 * z, sy + 40 * z);
        ctx.stroke();

        // PNP arrow points INWARD toward base bar
        drawArrow(ctx, sx + 14 * z, sy + 24 * z, sx - 4 * z, sy + 8 * z, z);

        drawPinDot(ctx, sx - 20 * z, sy, z);
        drawPinDot(ctx, sx + 20 * z, sy - 40 * z, z);
        drawPinDot(ctx, sx + 20 * z, sy + 40 * z, z);
    }

    // ═══════════════════════════════════════════
    // NEW SYMBOL RENDERERS — epoch_40.pt classes
    // ═══════════════════════════════════════════

    function drawVSS(ctx, sx, sy, z) {
        // VSS is a negative supply voltage source — drawn like DC source with V- label
        const r = 18 * z;

        // Top lead (pin at [0,-40])
        ctx.beginPath();
        ctx.moveTo(sx, sy - 40 * z);
        ctx.lineTo(sx, sy - r);
        ctx.stroke();

        // Circle
        ctx.beginPath();
        ctx.arc(sx, sy, r, 0, Math.PI * 2);
        ctx.stroke();

        // V- label inside
        ctx.fillStyle = "#E0E0E0";
        ctx.font = `bold ${11 * z}px 'Segoe UI', Arial`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("V−", sx, sy);

        // Bottom lead (pin at [0,40])
        ctx.beginPath();
        ctx.moveTo(sx, sy + r);
        ctx.lineTo(sx, sy + 40 * z);
        ctx.stroke();

        drawPinDot(ctx, sx, sy - 40 * z, z);
        drawPinDot(ctx, sx, sy + 40 * z, z);
    }

    function drawCapacitorPolarized(ctx, sx, sy, z) {
        const gap = 4 * z;
        const plateH = 16 * z;

        // Left lead
        ctx.beginPath();
        ctx.moveTo(sx - 40 * z, sy);
        ctx.lineTo(sx - gap, sy);
        ctx.stroke();

        // Left plate (straight — positive side)
        ctx.beginPath();
        ctx.moveTo(sx - gap, sy - plateH);
        ctx.lineTo(sx - gap, sy + plateH);
        ctx.stroke();

        // Right plate (curved — negative side)
        ctx.beginPath();
        ctx.arc(sx + gap + 12 * z, sy, 14 * z, Math.PI * 0.65, Math.PI * 1.35);
        ctx.stroke();

        // Right lead
        ctx.beginPath();
        ctx.moveTo(sx + gap + 2 * z, sy);
        ctx.lineTo(sx + 40 * z, sy);
        ctx.stroke();

        // Plus sign near positive plate
        ctx.fillStyle = "#FF9800";
        ctx.font = `bold ${10 * z}px 'Segoe UI', Arial`;
        ctx.textAlign = "center";
        ctx.textBaseline = "bottom";
        ctx.fillText("+", sx - gap - 6 * z, sy - plateH);

        drawPinDot(ctx, sx - 40 * z, sy, z);
        drawPinDot(ctx, sx + 40 * z, sy, z);
    }

    function drawResistorPhoto(ctx, sx, sy, z) {
        // Draw standard resistor first
        drawResistor(ctx, sx, sy, z);

        // Add light arrows (two diagonal arrows pointing at the body)
        ctx.strokeStyle = "#FFD54F";
        ctx.lineWidth = Math.max(1, 1.5 * z);
        // Arrow 1
        ctx.beginPath();
        ctx.moveTo(sx - 14 * z, sy - 18 * z);
        ctx.lineTo(sx - 4 * z, sy - 10 * z);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(sx - 4 * z, sy - 10 * z);
        ctx.lineTo(sx - 10 * z, sy - 10 * z);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(sx - 4 * z, sy - 10 * z);
        ctx.lineTo(sx - 4 * z, sy - 16 * z);
        ctx.stroke();
        // Arrow 2
        ctx.beginPath();
        ctx.moveTo(sx + 0 * z, sy - 18 * z);
        ctx.lineTo(sx + 10 * z, sy - 10 * z);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(sx + 10 * z, sy - 10 * z);
        ctx.lineTo(sx + 4 * z, sy - 10 * z);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(sx + 10 * z, sy - 10 * z);
        ctx.lineTo(sx + 10 * z, sy - 16 * z);
        ctx.stroke();
        // Restore stroke color
        ctx.strokeStyle = "#E0E0E0";
        ctx.lineWidth = Math.max(1, 2 * z);
    }

    function drawDiodeLED(ctx, sx, sy, z) {
        // Draw standard diode first
        drawDiode(ctx, sx, sy, z);

        // Add emission arrows (pointing away from the diode)
        ctx.strokeStyle = "#FFD54F";
        ctx.lineWidth = Math.max(1, 1.5 * z);
        // Arrow 1
        ctx.beginPath();
        ctx.moveTo(sx + 4 * z, sy - 12 * z);
        ctx.lineTo(sx + 14 * z, sy - 20 * z);
        ctx.stroke();
        // Arrowhead
        ctx.beginPath();
        ctx.moveTo(sx + 14 * z, sy - 20 * z);
        ctx.lineTo(sx + 8 * z, sy - 19 * z);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(sx + 14 * z, sy - 20 * z);
        ctx.lineTo(sx + 13 * z, sy - 14 * z);
        ctx.stroke();
        // Arrow 2
        ctx.beginPath();
        ctx.moveTo(sx + 10 * z, sy - 10 * z);
        ctx.lineTo(sx + 20 * z, sy - 18 * z);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(sx + 20 * z, sy - 18 * z);
        ctx.lineTo(sx + 14 * z, sy - 17 * z);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(sx + 20 * z, sy - 18 * z);
        ctx.lineTo(sx + 19 * z, sy - 12 * z);
        ctx.stroke();
        // Restore
        ctx.strokeStyle = "#E0E0E0";
        ctx.lineWidth = Math.max(1, 2 * z);
    }

    function drawDiodeZener(ctx, sx, sy, z) {
        const triW = 14 * z;
        const triH = 10 * z;

        // Left lead
        ctx.beginPath();
        ctx.moveTo(sx - 40 * z, sy);
        ctx.lineTo(sx - triW, sy);
        ctx.stroke();

        // Triangle (anode)
        ctx.beginPath();
        ctx.moveTo(sx - triW, sy - triH);
        ctx.lineTo(sx - triW, sy + triH);
        ctx.lineTo(sx + triW, sy);
        ctx.closePath();
        ctx.stroke();

        // Zener cathode bar with bent ends
        ctx.beginPath();
        ctx.moveTo(sx + triW - 4 * z, sy - triH - 3 * z);  // top bend
        ctx.lineTo(sx + triW, sy - triH);
        ctx.lineTo(sx + triW, sy + triH);
        ctx.lineTo(sx + triW + 4 * z, sy + triH + 3 * z);  // bottom bend
        ctx.stroke();

        // Right lead
        ctx.beginPath();
        ctx.moveTo(sx + triW, sy);
        ctx.lineTo(sx + 40 * z, sy);
        ctx.stroke();

        drawPinDot(ctx, sx - 40 * z, sy, z);
        drawPinDot(ctx, sx + 40 * z, sy, z);
    }

    function drawMOSFET(ctx, sx, sy, z) {
        // Gate lead (pin at [-20, 0])
        ctx.beginPath();
        ctx.moveTo(sx - 20 * z, sy);
        ctx.lineTo(sx - 6 * z, sy);
        ctx.stroke();

        // Gate vertical bar (insulated)
        ctx.lineWidth = Math.max(1, 2.5 * z);
        ctx.beginPath();
        ctx.moveTo(sx - 6 * z, sy - 14 * z);
        ctx.lineTo(sx - 6 * z, sy + 14 * z);
        ctx.stroke();
        ctx.lineWidth = Math.max(1, 2 * z);

        // Channel bar (dashed for enhancement mode)
        ctx.beginPath();
        ctx.moveTo(sx - 2 * z, sy - 14 * z);
        ctx.lineTo(sx - 2 * z, sy + 14 * z);
        ctx.stroke();

        // Drain lead (top — pin at [20, -40])
        ctx.beginPath();
        ctx.moveTo(sx - 2 * z, sy - 10 * z);
        ctx.lineTo(sx + 10 * z, sy - 10 * z);
        ctx.lineTo(sx + 10 * z, sy - 24 * z);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(sx + 10 * z, sy - 24 * z);
        ctx.lineTo(sx + 20 * z, sy - 40 * z);
        ctx.stroke();

        // Source lead (bottom — pin at [20, 40])
        ctx.beginPath();
        ctx.moveTo(sx - 2 * z, sy + 10 * z);
        ctx.lineTo(sx + 10 * z, sy + 10 * z);
        ctx.lineTo(sx + 10 * z, sy + 24 * z);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(sx + 10 * z, sy + 24 * z);
        ctx.lineTo(sx + 20 * z, sy + 40 * z);
        ctx.stroke();

        // Arrow on source (pointing inward for N-channel)
        drawArrow(ctx, sx + 10 * z, sy + 10 * z, sx - 2 * z, sy + 10 * z, z);

        // Body connection (substrate to source)
        ctx.beginPath();
        ctx.moveTo(sx - 2 * z, sy);
        ctx.lineTo(sx + 10 * z, sy);
        ctx.lineTo(sx + 10 * z, sy + 10 * z);
        ctx.stroke();

        drawPinDot(ctx, sx - 20 * z, sy, z);
        drawPinDot(ctx, sx + 20 * z, sy - 40 * z, z);
        drawPinDot(ctx, sx + 20 * z, sy + 40 * z, z);
    }

    function drawPhototransistor(ctx, sx, sy, z) {
        // Draw standard NPN BJT
        drawBJT_NPN(ctx, sx, sy, z);

        // Add incoming light arrows
        ctx.strokeStyle = "#FFD54F";
        ctx.lineWidth = Math.max(1, 1.5 * z);
        // Arrow 1
        ctx.beginPath();
        ctx.moveTo(sx - 24 * z, sy - 24 * z);
        ctx.lineTo(sx - 12 * z, sy - 12 * z);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(sx - 12 * z, sy - 12 * z);
        ctx.lineTo(sx - 18 * z, sy - 12 * z);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(sx - 12 * z, sy - 12 * z);
        ctx.lineTo(sx - 12 * z, sy - 18 * z);
        ctx.stroke();
        // Arrow 2
        ctx.beginPath();
        ctx.moveTo(sx - 18 * z, sy - 18 * z);
        ctx.lineTo(sx - 6 * z, sy - 6 * z);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(sx - 6 * z, sy - 6 * z);
        ctx.lineTo(sx - 12 * z, sy - 6 * z);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(sx - 6 * z, sy - 6 * z);
        ctx.lineTo(sx - 6 * z, sy - 12 * z);
        ctx.stroke();
        // Restore
        ctx.strokeStyle = "#E0E0E0";
        ctx.lineWidth = Math.max(1, 2 * z);
    }

    function drawOpAmp(ctx, sx, sy, z) {
        // Triangle body
        ctx.beginPath();
        ctx.moveTo(sx - 24 * z, sy - 30 * z);
        ctx.lineTo(sx - 24 * z, sy + 30 * z);
        ctx.lineTo(sx + 24 * z, sy);
        ctx.closePath();
        ctx.stroke();

        // V+ input lead (pin at [-30, -20])
        ctx.beginPath();
        ctx.moveTo(sx - 30 * z, sy - 20 * z);
        ctx.lineTo(sx - 24 * z, sy - 20 * z);
        ctx.stroke();

        // V− input lead (pin at [-30, 20])
        ctx.beginPath();
        ctx.moveTo(sx - 30 * z, sy + 20 * z);
        ctx.lineTo(sx - 24 * z, sy + 20 * z);
        ctx.stroke();

        // Output lead (pin at [30, 0])
        ctx.beginPath();
        ctx.moveTo(sx + 24 * z, sy);
        ctx.lineTo(sx + 30 * z, sy);
        ctx.stroke();

        // + sign at non-inverting input
        ctx.fillStyle = "#4FC1FF";
        ctx.font = `bold ${12 * z}px 'Segoe UI', Arial`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("+", sx - 16 * z, sy - 16 * z);

        // − sign at inverting input
        ctx.fillText("−", sx - 16 * z, sy + 16 * z);

        drawPinDot(ctx, sx - 30 * z, sy - 20 * z, z);
        drawPinDot(ctx, sx - 30 * z, sy + 20 * z, z);
        drawPinDot(ctx, sx + 30 * z, sy, z);
    }

    function drawIC(ctx, sx, sy, z) {
        const w = 60 * z;
        const h = 36 * z;

        // Rectangle body
        ctx.strokeRect(sx - w / 2, sy - h / 2, w, h);

        // IC label
        ctx.fillStyle = "#E0E0E0";
        ctx.font = `bold ${11 * z}px 'Segoe UI', Arial`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("IC", sx, sy);

        // Left lead (pin at [-40, 0])
        ctx.beginPath();
        ctx.moveTo(sx - 40 * z, sy);
        ctx.lineTo(sx - w / 2, sy);
        ctx.stroke();

        // Right lead (pin at [40, 0])
        ctx.beginPath();
        ctx.moveTo(sx + w / 2, sy);
        ctx.lineTo(sx + 40 * z, sy);
        ctx.stroke();

        // Notch at top center
        ctx.beginPath();
        ctx.arc(sx, sy - h / 2, 4 * z, 0, Math.PI);
        ctx.stroke();

        drawPinDot(ctx, sx - 40 * z, sy, z);
        drawPinDot(ctx, sx + 40 * z, sy, z);
    }

    function drawTransformer(ctx, sx, sy, z) {
        // Primary coil (left side) — 3 humps
        const humps = 3;
        const humpH = 8 * z;
        const totalH = humps * humpH * 2;
        const startY = sy - totalH / 2;

        // Top-left lead (pin at [-40, -20])
        ctx.beginPath();
        ctx.moveTo(sx - 40 * z, sy - 20 * z);
        ctx.lineTo(sx - 12 * z, sy - 20 * z);
        ctx.stroke();

        // Primary coil arcs
        for (let i = 0; i < humps; i++) {
            const cy = startY + i * humpH * 2 + humpH;
            ctx.beginPath();
            ctx.arc(sx - 12 * z, cy, humpH, -Math.PI / 2, Math.PI / 2, false);
            ctx.stroke();
        }

        // Bottom-left lead (pin at [-40, 20])
        ctx.beginPath();
        ctx.moveTo(sx - 12 * z, startY + totalH);
        ctx.lineTo(sx - 12 * z, sy + 20 * z);
        ctx.lineTo(sx - 40 * z, sy + 20 * z);
        ctx.stroke();

        // Core lines (two vertical parallel lines)
        ctx.beginPath();
        ctx.moveTo(sx - 4 * z, sy - 24 * z);
        ctx.lineTo(sx - 4 * z, sy + 24 * z);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(sx + 4 * z, sy - 24 * z);
        ctx.lineTo(sx + 4 * z, sy + 24 * z);
        ctx.stroke();

        // Secondary coil (right side) — 3 humps
        // Top-right lead (pin at [40, -20])
        ctx.beginPath();
        ctx.moveTo(sx + 40 * z, sy - 20 * z);
        ctx.lineTo(sx + 12 * z, sy - 20 * z);
        ctx.stroke();

        for (let i = 0; i < humps; i++) {
            const cy = startY + i * humpH * 2 + humpH;
            ctx.beginPath();
            ctx.arc(sx + 12 * z, cy, humpH, Math.PI / 2, -Math.PI / 2, false);
            ctx.stroke();
        }

        // Bottom-right lead (pin at [40, 20])
        ctx.beginPath();
        ctx.moveTo(sx + 12 * z, startY + totalH);
        ctx.lineTo(sx + 12 * z, sy + 20 * z);
        ctx.lineTo(sx + 40 * z, sy + 20 * z);
        ctx.stroke();

        drawPinDot(ctx, sx - 40 * z, sy - 20 * z, z);
        drawPinDot(ctx, sx - 40 * z, sy + 20 * z, z);
        drawPinDot(ctx, sx + 40 * z, sy - 20 * z, z);
        drawPinDot(ctx, sx + 40 * z, sy + 20 * z, z);
    }

    function drawJunction(ctx, sx, sy, z) {
        // Simple filled dot — wire junction marker
        ctx.fillStyle = "#4FC1FF";
        ctx.beginPath();
        ctx.arc(sx, sy, 4 * z, 0, Math.PI * 2);
        ctx.fill();
    }

    function drawCrossover(ctx, sx, sy, z) {
        // Small bump/arc showing wires cross without connecting
        const len = 10 * z;
        // Horizontal wire through
        ctx.beginPath();
        ctx.moveTo(sx - len, sy);
        ctx.lineTo(sx - 4 * z, sy);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(sx + 4 * z, sy);
        ctx.lineTo(sx + len, sy);
        ctx.stroke();
        // Arc (bridge) over center
        ctx.beginPath();
        ctx.arc(sx, sy - 3 * z, 5 * z, 0.3 * Math.PI, 0.7 * Math.PI, false);
        ctx.stroke();
    }

    function drawTerminal(ctx, sx, sy, z) {
        // Small open circle with a lead
        ctx.beginPath();
        ctx.arc(sx, sy, 5 * z, 0, Math.PI * 2);
        ctx.stroke();

        drawPinDot(ctx, sx, sy, z);
    }

    function drawArrow(ctx, fromX, fromY, toX, toY, z) {
        const angle = Math.atan2(toY - fromY, toX - fromX);
        const len = 6 * z;
        const spread = 0.45;
        const midX = (fromX + toX) / 2;
        const midY = (fromY + toY) / 2;

        ctx.beginPath();
        ctx.moveTo(midX, midY);
        ctx.lineTo(midX - len * Math.cos(angle - spread), midY - len * Math.sin(angle - spread));
        ctx.moveTo(midX, midY);
        ctx.lineTo(midX - len * Math.cos(angle + spread), midY - len * Math.sin(angle + spread));
        ctx.stroke();
    }

    function drawPinDot(ctx, x, y, z) {
        ctx.fillStyle = "#4FC1FF";
        ctx.beginPath();
        ctx.arc(x, y, 2.5 * z, 0, Math.PI * 2);
        ctx.fill();
    }

    // Fallback renderer for unknown component types
    function drawFallback(ctx, sx, sy, z, type) {
        const size = 30 * z;
        ctx.strokeRect(sx - size / 2, sy - size / 2, size, size);
        ctx.fillStyle = "#E0E0E0";
        ctx.font = `${10 * z}px Arial`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(type, sx, sy);
    }

    // ═══════════════════════════════════════════
    // MAIN RENDERING ENGINE
    // ═══════════════════════════════════════════
    function render() {
        // Clear Canvas
        ctx.fillStyle = "#1E1E1E";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw Grid dots
        ctx.fillStyle = "#444444";
        const step = gridSize * zoom;
        if (step > 5) {
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
            if (wire.length < 2) return;
            const p1 = worldToScreen(wire[0].x, wire[0].y);
            const p2 = worldToScreen(wire[1].x, wire[1].y);
            
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            // Force orthogonal routing visually if diagonal
            if (wire[0].x !== wire[1].x && wire[0].y !== wire[1].y) {
                ctx.lineTo(p2.x, p1.y);
            }
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();

            // Wire endpoint dots
            ctx.fillStyle = "#4FC1FF";
            ctx.beginPath(); ctx.arc(p1.x, p1.y, 3 * zoom, 0, Math.PI * 2); ctx.fill();
            ctx.beginPath(); ctx.arc(p2.x, p2.y, 3 * zoom, 0, Math.PI * 2); ctx.fill();
        });

        // Draw Components
        components.forEach(comp => {
            const SKIP_RENDER_TYPES = ['wire', 'junction', 'crossover', 'terminal', 'text'];
            if (SKIP_RENDER_TYPES.includes(comp.type)) return;
            const pos = worldToScreen(comp.x, comp.y);

            if (comp.type === 'label') {
                const labelText = comp.params && comp.params.name ? comp.params.name : comp.name;
                ctx.fillStyle = "#FFC107";
                ctx.font = `bold ${14 * zoom}px 'Segoe UI', Arial`;
                ctx.textAlign = "center";
                ctx.textBaseline = "bottom";
                ctx.fillText(labelText, pos.x, pos.y - 8 * zoom);
                return;
            }
            // Selection highlight
            if (comp === selectedComp) {
                const db = COMPONENT_DB[comp.type];
                const hb = db ? db.hitbox : { w: 40, h: 40 };
                ctx.strokeStyle = "#0078D7";
                ctx.lineWidth = 2;
                ctx.setLineDash([4, 3]);
                ctx.strokeRect(
                    pos.x - (hb.w / 2 + 6) * zoom,
                    pos.y - (hb.h / 2 + 6) * zoom,
                    (hb.w + 12) * zoom,
                    (hb.h + 12) * zoom
                );
                ctx.setLineDash([]);
            }

            // Draw the schematic symbol
            ctx.strokeStyle = "#E0E0E0";
            ctx.lineWidth = Math.max(1, 2 * zoom);
            const renderer = SYMBOL_RENDERERS[comp.type];
            if (renderer) {
                renderer(ctx, pos.x, pos.y, zoom);
            } else {
                drawFallback(ctx, pos.x, pos.y, zoom, comp.type);
            }

            // Draw component name label (above)
            ctx.fillStyle = "#E0E0E0";
            ctx.font = `bold ${12 * zoom}px 'Segoe UI', Arial`;
            ctx.textAlign = "center";
            ctx.textBaseline = "bottom";
            const db = COMPONENT_DB[comp.type];
            const labelOffY = db ? db.hitbox.h / 2 + 14 : 30;
            ctx.fillText(comp.name, pos.x, pos.y - labelOffY * zoom);

            // Draw value label (below)
            const mainValue = comp.value || getDisplayValue(comp);
            if (mainValue && comp.type !== 'ground') {
                ctx.fillStyle = "#FF9800";
                ctx.font = `${11 * zoom}px 'Segoe UI', Arial`;
                ctx.textBaseline = "top";
                ctx.fillText(mainValue, pos.x, pos.y + labelOffY * zoom);
            }
        });

        // Draw pin indicators in wire mode (show all available pins)
        if (mode === 'wire') {
            ctx.save();
            components.forEach(comp => {
                const pins = getCompPins(comp);
                pins.forEach(pin => {
                    const sp = worldToScreen(pin.x, pin.y);
                    // Highlight the hovered pin with a larger bright ring
                    if (hoveredPin && hoveredPin.x === pin.x && hoveredPin.y === pin.y) {
                        ctx.fillStyle = "rgba(0, 255, 136, 0.8)";
                        ctx.strokeStyle = "rgba(0, 255, 136, 0.9)";
                        ctx.lineWidth = 2;
                        ctx.beginPath();
                        ctx.arc(sp.x, sp.y, 7 * zoom, 0, Math.PI * 2);
                        ctx.stroke();
                        ctx.beginPath();
                        ctx.arc(sp.x, sp.y, 3.5 * zoom, 0, Math.PI * 2);
                        ctx.fill();
                    } else {
                        // Subtle pin markers for unconnected pins
                        ctx.fillStyle = "rgba(79, 193, 255, 0.3)";
                        ctx.beginPath();
                        ctx.arc(sp.x, sp.y, 4 * zoom, 0, Math.PI * 2);
                        ctx.fill();
                    }
                });
            });
            ctx.restore();
        }

        // Ghost wire preview (while routing)
        if (mode === 'wire' && wireStart) {
            const p1 = worldToScreen(wireStart.x, wireStart.y);
            const p2 = worldToScreen(mousePos.x, wireStart.y);
            const p3 = worldToScreen(mousePos.x, mousePos.y);

            ctx.strokeStyle = "rgba(79, 193, 255, 0.5)";
            ctx.setLineDash([5, 5]);
            ctx.lineWidth = Math.max(1, 2 * zoom);
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.lineTo(p3.x, p3.y);
            ctx.stroke();
            ctx.setLineDash([]);

            // Snap-target dot at the ghost endpoint
            const endDot = worldToScreen(mousePos.x, mousePos.y);
            ctx.fillStyle = hoveredPin ? "#00FF88" : "#4FC1FF";
            ctx.beginPath();
            ctx.arc(endDot.x, endDot.y, 4 * zoom, 0, Math.PI * 2);
            ctx.fill();
        }

        // Ghost component preview (while placing)
        if (mode !== 'select' && mode !== 'wire') {
            const gPos = worldToScreen(mousePos.x, mousePos.y);
            ctx.globalAlpha = 0.4;
            ctx.strokeStyle = "#E0E0E0";
            ctx.lineWidth = Math.max(1, 2 * zoom);
            const renderer = SYMBOL_RENDERERS[mode];
            if (renderer) {
                renderer(ctx, gPos.x, gPos.y, zoom);
            }
            ctx.globalAlpha = 1.0;
        }
    }

    function getDisplayValue(comp) {
        if (!comp.params) return '';
        if (comp.params.value) return comp.params.value;
        if (comp.params.dc) return comp.params.dc;
        if (comp.params.mag) return comp.params.mag;
        if (comp.params.model) return comp.params.model;
        return '';
    }

    // ═══════════════════════════════════════════
    // PROPERTIES INSPECTOR (Data-Binding)
    // ═══════════════════════════════════════════
    function updatePropertiesPanel() {
        const panel = document.getElementById("propertiesPanel");
        panel.innerHTML = '';

        if (!selectedComp) {
            panel.innerHTML = '<p class="placeholder-text">Select a component</p>';
            return;
        }

        const comp = selectedComp;
        const db = COMPONENT_DB[comp.type];

        // Type badge
        const badge = document.createElement('div');
        badge.className = 'prop-type-badge';
        badge.textContent = db ? db.label : comp.type;
        panel.appendChild(badge);

        // Section: Identity
        const idTitle = document.createElement('p');
        idTitle.className = 'prop-section-title';
        idTitle.textContent = 'Identity';
        panel.appendChild(idTitle);

        // Name field
        addPropField(panel, 'Name', comp.name, (val) => {
            comp.name = val;
            render();
        });

        // Type field (read-only)
        addPropField(panel, 'Type', comp.type, null, true);

        // Position fields (read-only)
        addPropField(panel, 'X', String(comp.x), null, true);
        addPropField(panel, 'Y', String(comp.y), null, true);

        // Section: Parameters
        if (comp.params && Object.keys(comp.params).length > 0) {
            const paramTitle = document.createElement('p');
            paramTitle.className = 'prop-section-title';
            paramTitle.textContent = 'SPICE Parameters';
            panel.appendChild(paramTitle);

            for (const [key, val] of Object.entries(comp.params)) {
                addPropField(panel, key, String(val), (newVal) => {
                    comp.params[key] = newVal;
                    // Sync the top-level value field
                    if (key === 'value' || key === 'dc' || key === 'mag') {
                        comp.value = newVal;
                    }
                    render();
                });
            }
        }

        // Delete button
        const delBtn = document.createElement('button');
        delBtn.className = 'prop-delete-btn';
        delBtn.textContent = '🗑 Delete Component';
        delBtn.addEventListener('click', () => {
            const idx = components.indexOf(comp);
            if (idx !== -1) components.splice(idx, 1);
            selectedComp = null;
            updatePropertiesPanel();
            render();
        });
        panel.appendChild(delBtn);
    }

    function addPropField(panel, label, value, onChange, readOnly = false) {
        const row = document.createElement('div');
        row.className = 'prop-row';

        const lbl = document.createElement('span');
        lbl.className = 'prop-label';
        lbl.textContent = label;
        row.appendChild(lbl);

        const input = document.createElement('input');
        input.className = 'prop-input';
        input.type = 'text';
        input.value = value;
        if (readOnly) {
            input.readOnly = true;
        } else if (onChange) {
            input.addEventListener('input', (e) => onChange(e.target.value));
        }
        row.appendChild(input);

        panel.appendChild(row);
    }

    // ═══════════════════════════════════════════
    // TOOLBAR LOGIC
    // ═══════════════════════════════════════════
    document.querySelectorAll('.btn-tool').forEach(btn => {
        btn.addEventListener('click', (e) => {
            mode = e.target.id.replace('tool-', '');
            wireStart = null;
            updateToolUI();
            
            // Reset select dropdown if a standard button was clicked
            const selectEl = document.getElementById('tool-source-select');
            if (selectEl) selectEl.value = '';
            
            render();
        });
    });

    const sourceSelect = document.getElementById('tool-source-select');
    if (sourceSelect) {
        sourceSelect.addEventListener('change', (e) => {
            mode = e.target.value;
            wireStart = null;
            updateToolUI();
            render();
        });
    }

    function updateToolUI() {
        document.querySelectorAll('.btn-tool').forEach(btn => btn.classList.remove('active'));
        const activeBtn = document.getElementById(`tool-${mode}`);
        if (activeBtn) activeBtn.classList.add('active');
        
        // Highlight select dropdown if a source is active
        const sourceSelect = document.getElementById('tool-source-select');
        if (sourceSelect && sourceSelect.value === mode && mode !== '') {
            sourceSelect.style.color = 'var(--accent-cyan)';
        } else if (sourceSelect) {
            sourceSelect.style.color = '';
        }
    }

    // ═══════════════════════════════════════════
    // NETLIST PANEL TOGGLE
    // ═══════════════════════════════════════════
    const netlistToggle = document.getElementById("netlistToggle");
    if (netlistToggle) {
        netlistToggle.addEventListener('click', () => {
            const panel = document.getElementById("netlistPreview");
            panel.classList.toggle('collapsed');
            const icon = netlistToggle.querySelector('.toggle-icon');
            icon.textContent = panel.classList.contains('collapsed') ? '▶' : '▼';
        });
    }

    // ═══════════════════════════════════════════
    // SIMULATION SETUP MODAL
    // (Matches PySpice Studio SimulationDialog)
    // ═══════════════════════════════════════════
    const simModal = document.getElementById('simModal');
    const simModeSelect = document.getElementById('simModeSelect');
    const simParamsContainer = document.getElementById('simParamsContainer');

    function openSimModal() {
        if (components.length === 0) {
            document.getElementById('statusText').innerText = '⚠️ No components to simulate.';
            return;
        }
        // Restore last-used mode
        simModeSelect.value = simConfig.mode;
        rebuildSimForm();
        rebuildSignalList();
        simModal.style.display = 'flex';
        // Solve nodes in the background to populate dropdowns
        solveNodesForDialog();
    }

    function closeSimModal() {
        simModal.style.display = 'none';
    }

    // Tab switching
    document.querySelectorAll('.modal-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.modal-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.modal-tab-content').forEach(c => c.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById(tab.dataset.tab).classList.add('active');
        });
    });

    // Mode change → rebuild form
    simModeSelect.addEventListener('change', () => {
        simConfig.mode = simModeSelect.value;
        rebuildSimForm();
    });

    // Open modal from Setup or RUN button
    document.getElementById('btnSimSetup').addEventListener('click', openSimModal);
    document.getElementById('btnSimulate').addEventListener('click', openSimModal);
    document.getElementById('modalClose').addEventListener('click', closeSimModal);
    document.getElementById('btnModalCancel').addEventListener('click', closeSimModal);

    // Close modal when clicking the overlay background
    simModal.addEventListener('click', (e) => {
        if (e.target === simModal) closeSimModal();
    });

    // ── Dynamic Form Builder (matching Python app's rebuild_form) ──
    function rebuildSimForm() {
        simParamsContainer.innerHTML = '';
        const m = simModeSelect.value;

        if (m === 'op') {
            simParamsContainer.innerHTML = '<p class="params-op-message">Operating Point analysis requires no additional parameters.</p>';
            return;
        }

        if (m === 'tran') {
            addParamField('step', 'Step Time', simConfig.params.step || '0.1m');
            addParamField('stop', 'Stop Time', simConfig.params.stop || '80m');
            addParamField('start', 'Start Time', simConfig.params.start || '0');
        } else if (m === 'dc') {
            addParamSelect('source1', 'Source 1', availableSweepables, simConfig.params.source1 || '');
            addParamField('start', 'Start', simConfig.params.start || '0');
            addParamField('stop', 'Stop', simConfig.params.stop || '5');
            addParamField('incr', 'Increment', simConfig.params.incr || '0.1');
            // Separator
            const sep = document.createElement('hr');
            sep.className = 'params-separator';
            simParamsContainer.appendChild(sep);
            const sepLabel = document.createElement('p');
            sepLabel.className = 'params-separator-label';
            sepLabel.textContent = '— Secondary Sweep (Optional) —';
            simParamsContainer.appendChild(sepLabel);
            addParamSelect('source2', 'Source 2', ['None', ...availableSweepables], simConfig.params.source2 || 'None');
            addParamField('start2', 'Start 2', simConfig.params.start2 || '0');
            addParamField('stop2', 'Stop 2', simConfig.params.stop2 || '5');
            addParamField('incr2', 'Incr 2', simConfig.params.incr2 || '1');
        } else if (m === 'ac') {
            addParamSelect('type', 'Type', ['DEC', 'LIN'], simConfig.params.type || 'DEC');
            addParamField('points', 'Points', simConfig.params.points || '10');
            addParamField('fstart', 'Start Freq', simConfig.params.fstart || '1');
            addParamField('fstop', 'Stop Freq', simConfig.params.fstop || '10meg');
        }
    }

    function addParamField(key, label, defaultVal) {
        const row = document.createElement('div');
        row.className = 'modal-form-row';
        row.innerHTML = `
            <label class="modal-label">${label}</label>
            <input class="modal-input" type="text" data-param-key="${key}" value="${defaultVal}">
        `;
        row.querySelector('input').addEventListener('input', (e) => {
            simConfig.params[key] = e.target.value;
        });
        simParamsContainer.appendChild(row);
    }

    function addParamSelect(key, label, options, defaultVal) {
        const row = document.createElement('div');
        row.className = 'modal-form-row';
        const optHtml = options.map(o =>
            `<option value="${o}"${o === defaultVal ? ' selected' : ''}>${o}</option>`
        ).join('');
        row.innerHTML = `
            <label class="modal-label">${label}</label>
            <select class="modal-select" data-param-key="${key}">${optHtml}</select>
        `;
        row.querySelector('select').addEventListener('change', (e) => {
            simConfig.params[key] = e.target.value;
        });
        simParamsContainer.appendChild(row);
    }

    // ── Node Solving (pre-populates dropdowns) ──
    async function solveNodesForDialog() {
        try {
            const payload = {
                components: components.map(c => ({
                    type: c.type, name: c.name,
                    x: c.x, y: c.y,
                    params: c.params || {}, rotation: 0
                })),
                wires: wires.map(w => [
                    { x: w[0].x, y: w[0].y },
                    { x: w[1].x, y: w[1].y }
                ]),
                simConfig: { mode: 'op', params: {} }
            };

            const resp = await fetch('http://127.0.0.1:8000/api/solve_nodes', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await resp.json();

            if (data.status === 'success') {
                availableNodes = data.nodes || [];
                availableSources = data.sources || [];
                availableSweepables = data.sweepables || [];
                // Refresh dropdowns
                populateSelect('sigNodeSelect', availableNodes);
                populateSelect('sigSourceSelect', availableSources);
                // Rebuild form to refresh sweepable selects
                rebuildSimForm();
            }
        } catch (e) {
            console.warn('Could not solve nodes:', e);
        }
    }

    function populateSelect(selectId, options) {
        const sel = document.getElementById(selectId);
        if (!sel) return;
        sel.innerHTML = '';
        if (options.length === 0) {
            sel.innerHTML = '<option value="">— none —</option>';
            return;
        }
        options.forEach(opt => {
            const o = document.createElement('option');
            o.value = opt; o.textContent = opt;
            sel.appendChild(o);
        });
    }

    // ── Signal List Management ──
    document.getElementById('btnAddVoltage').addEventListener('click', () => {
        const node = document.getElementById('sigNodeSelect').value;
        if (!node) return;
        const sig = `v(${node})`;
        const color = document.getElementById('sigColorSelect').value;
        const win = document.getElementById('sigWindowSelect').value;
        plotSignals.push({ signal: sig, color, window: win });
        rebuildSignalList();
    });

    document.getElementById('btnAddCurrent').addEventListener('click', () => {
        const src = document.getElementById('sigSourceSelect').value;
        if (!src) return;
        const sig = `i(${src})`;
        const color = document.getElementById('sigColorSelect').value;
        const win = document.getElementById('sigWindowSelect').value;
        plotSignals.push({ signal: sig, color, window: win });
        rebuildSignalList();
    });

    function rebuildSignalList() {
        const list = document.getElementById('signalList');
        const noMsg = document.getElementById('noSignalsMsg');
        list.innerHTML = '';

        if (plotSignals.length === 0) {
            list.innerHTML = '<p class="placeholder-text" id="noSignalsMsg">No signals added yet</p>';
            return;
        }

        plotSignals.forEach((sig, idx) => {
            const item = document.createElement('div');
            item.className = 'signal-item';
            item.innerHTML = `
                <div style="display:flex;align-items:center;gap:6px;">
                    <span class="signal-color-dot" style="background:${sig.color}"></span>
                    <span class="signal-item-label">${sig.signal}</span>
                    <span class="signal-item-meta">Win ${sig.window}</span>
                </div>
                <button class="signal-remove-btn" data-idx="${idx}">&times;</button>
            `;
            item.querySelector('.signal-remove-btn').addEventListener('click', () => {
                plotSignals.splice(idx, 1);
                rebuildSignalList();
            });
            list.appendChild(item);
        });
    }

    // ── Build final sim config from dialog state ──
    function collectSimConfig() {
        // Collect params from form inputs
        const paramInputs = simParamsContainer.querySelectorAll('[data-param-key]');
        paramInputs.forEach(el => {
            simConfig.params[el.dataset.paramKey] = el.value;
        });

        // Build plots dict: window_id → [signals]
        simConfig.plots = {};
        // Build colors dict
        const bgColor = document.getElementById('plotBgColor').value;
        simConfig.colors = {
            '0': bgColor,
            '1': (bgColor === 'black' || bgColor === '#1E1E1E') ? 'white' : 'black'
        };

        let colorIdx = 2;
        plotSignals.forEach(sig => {
            if (!simConfig.plots[sig.window]) {
                simConfig.plots[sig.window] = [];
            }
            simConfig.plots[sig.window].push(sig.signal);
            simConfig.colors[String(colorIdx)] = sig.color;
            colorIdx++;
        });

        simConfig.mode = simModeSelect.value;
        return simConfig;
    }

    // ═══════════════════════════════════════════
    // SIMULATION EXECUTION
    // ═══════════════════════════════════════════
    document.getElementById('btnModalRun').addEventListener('click', async () => {
        const config = collectSimConfig();
        closeSimModal();
        await runSimulation(config);
    });

    async function runSimulation(config) {
        const statusText = document.getElementById('statusText');
        const netlistEl = document.getElementById('netlistText');
        const simOutput = document.getElementById('simulationOutput');

        statusText.innerText = '⚙️ Generating netlist & running simulation...';
        simOutput.innerHTML = '<p class="placeholder-text">Running simulation...</p>';

        const payload = {
            components: components.map(c => ({
                type: c.type, name: c.name,
                x: c.x, y: c.y,
                params: c.params || {}, rotation: 0
            })),
            wires: wires.map(w => [
                { x: w[0].x, y: w[0].y },
                { x: w[1].x, y: w[1].y }
            ]),
            simConfig: config
        };

        try {
            const response = await fetch('http://127.0.0.1:8000/api/simulate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await response.json();

            if (data.status === 'success') {
                statusText.innerText = '✅ Simulation complete.';
                netlistEl.textContent = data.netlist || 'No netlist generated';

                // Update available nodes/sources from response
                if (data.nodes) availableNodes = data.nodes;
                if (data.sources) availableSources = data.sources;
                if (data.sweepables) availableSweepables = data.sweepables;

                // Build output display
                let outputHtml = '';

                // Show plot images first
                if (data.plot_images && data.plot_images.length > 0) {
                    data.plot_images.forEach((imgUrl, idx) => {
                        const fullUrl = `http://127.0.0.1:8000${imgUrl}?t=${Date.now()}`;
                        outputHtml += `
                            <div class="plot-image-container">
                                <img src="${fullUrl}" alt="Plot Window ${idx + 1}"
                                     onclick="openLightbox(this.src)">
                                <p class="plot-image-label">Graph Window ${idx + 1}</p>
                            </div>
                        `;
                    });
                }

                // Show raw output (collapsed if plots exist)
                if (data.raw_output) {
                    if (data.plot_images && data.plot_images.length > 0) {
                        outputHtml += `<details style="margin-top:8px;"><summary style="cursor:pointer;color:var(--text-secondary);font-size:12px;">Raw ngspice output</summary><pre>${escapeHtml(data.raw_output)}</pre></details>`;
                    } else {
                        outputHtml += `<pre>${escapeHtml(data.raw_output)}</pre>`;
                    }
                }

                if (!outputHtml) {
                    outputHtml = '<p class="sim-success">Simulation completed successfully.</p>';
                }

                simOutput.innerHTML = outputHtml;
            } else {
                statusText.innerText = '❌ Simulation failed.';
                netlistEl.textContent = data.netlist || 'Error';
                simOutput.innerHTML = `<p class="sim-error">${escapeHtml(data.message || 'Unknown error')}</p>`;
                if (data.raw_output) {
                    simOutput.innerHTML += `<pre>${escapeHtml(data.raw_output)}</pre>`;
                }
            }
        } catch (err) {
            console.error('Simulation request failed:', err);
            statusText.innerText = '❌ Connection error.';
            simOutput.innerHTML = `<p class="sim-error">Could not reach backend at http://127.0.0.1:8000. Is the server running?</p>`;
        }
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // ═══════════════════════════════════════════
    // LIGHTBOX (Full-screen plot image viewer)
    // ═══════════════════════════════════════════
    // Exposed globally for onclick in dynamic HTML
    window.openLightbox = function(src) {
        const overlay = document.createElement('div');
        overlay.className = 'lightbox-overlay';
        overlay.innerHTML = `<img src="${src}" alt="Plot">`;
        overlay.addEventListener('click', () => overlay.remove());
        document.body.appendChild(overlay);
    };

    // ═══════════════════════════════════════════
    // A* ROUTER FOR IMPORT
    // ═══════════════════════════════════════════
    function routeAStar(start, goal, componentsList, gSize) {
        if (start.x === goal.x && start.y === goal.y) return [];
        
        const getObstacleAt = (x, y) => {
            if (Math.abs(x - start.x) < 1 && Math.abs(y - start.y) < 1) return false;
            if (Math.abs(x - goal.x) < 1 && Math.abs(y - goal.y) < 1) return false;
            
            for (const c of componentsList) {
                const db = COMPONENT_DB[c.type];
                const hb = db ? db.hitbox : { w: 40, h: 40 };
                // Inflate slightly to prevent wires grazing the bounding box
                const hw = (hb.w / 2) + gSize * 0.5;
                const hh = (hb.h / 2) + gSize * 0.5;
                if (x > c.x - hw && x < c.x + hw && y > c.y - hh && y < c.y + hh) {
                    return true;
                }
            }
            return false;
        };

        const openSet = [];
        const openMap = new Map();
        const closedSet = new Set();
        
        const startNode = {
            x: start.x, y: start.y,
            g: 0, f: 0,
            dir: null, parent: null
        };
        const key = (x, y) => `${x},${y}`;
        const h = (x, y) => Math.abs(x - goal.x) + Math.abs(y - goal.y);
        
        openSet.push(startNode);
        openMap.set(key(start.x, start.y), startNode);
        
        const dirs = [
            { dx: gSize, dy: 0, dir: 'R' },
            { dx: -gSize, dy: 0, dir: 'L' },
            { dx: 0, dy: gSize, dir: 'D' },
            { dx: 0, dy: -gSize, dir: 'U' }
        ];

        let bestGoal = null;
        let iter = 0;
        
        while (openSet.length > 0 && iter < 10000) {
            iter++;
            let minFIdx = 0;
            for (let i = 1; i < openSet.length; i++) {
                if (openSet[i].f < openSet[minFIdx].f) {
                    minFIdx = i;
                }
            }
            const curr = openSet.splice(minFIdx, 1)[0];
            const cKey = key(curr.x, curr.y);
            openMap.delete(cKey);
            closedSet.add(cKey);
            
            if (curr.x === goal.x && curr.y === goal.y) {
                bestGoal = curr;
                break;
            }
            
            for (const d of dirs) {
                const nx = curr.x + d.dx;
                const ny = curr.y + d.dy;
                const nKey = key(nx, ny);
                
                if (closedSet.has(nKey)) continue;
                if (getObstacleAt(nx, ny)) continue;
                
                // Heavy penalty for changing direction to encourage straight lines
                let turnCost = 0;
                if (curr.dir && curr.dir !== d.dir) turnCost = gSize * 3;
                
                const tentativeG = curr.g + gSize + turnCost;
                
                let neighbor = openMap.get(nKey);
                if (!neighbor) {
                    neighbor = { x: nx, y: ny, g: tentativeG, f: tentativeG + h(nx, ny), dir: d.dir, parent: curr };
                    openSet.push(neighbor);
                    openMap.set(nKey, neighbor);
                } else if (tentativeG < neighbor.g) {
                    neighbor.g = tentativeG;
                    neighbor.f = tentativeG + h(nx, ny);
                    neighbor.dir = d.dir;
                    neighbor.parent = curr;
                }
            }
        }
        
        if (!bestGoal) return null; // Pathfinding failed
        
        const path = [];
        let p = bestGoal;
        while (p) {
            path.push({ x: p.x, y: p.y });
            p = p.parent;
        }
        path.reverse();
        
        const segments = [];
        let segStart = path[0];
        for (let i = 1; i < path.length - 1; i++) {
            const prev = path[i-1];
            const curr = path[i];
            const next = path[i+1];
            
            const dx1 = curr.x - prev.x;
            const dy1 = curr.y - prev.y;
            const dx2 = next.x - curr.x;
            const dy2 = next.y - curr.y;
            
            if (Math.sign(dx1) !== Math.sign(dx2) || Math.sign(dy1) !== Math.sign(dy2)) {
                segments.push([{ x: segStart.x, y: segStart.y }, { x: curr.x, y: curr.y }]);
                segStart = curr;
            }
        }
        if (path.length > 1) {
            segments.push([{ x: segStart.x, y: segStart.y }, { x: path[path.length - 1].x, y: path[path.length - 1].y }]);
        }
        return segments;
    }

    // ═══════════════════════════════════════════
    // AI IMPORT LOGIC
    // ═══════════════════════════════════════════
    const fileInput = document.getElementById("fileInput");
    document.getElementById("btnLoadImage").addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        document.getElementById("statusText").innerText = "🤖 Processing Image...";
        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch("http://127.0.0.1:8000/api/detect", { method: "POST", body: formData });
            const data = await response.json();

            if (data.status === "success") {
                document.getElementById("statusText").innerText = `✅ Loaded ${data.components.length} components. Waiting for approval.`;
                
                // Show modal
                const modal = document.getElementById("aiDebugModal");
                const img = document.getElementById("debugPreviewImage");
                if (data.debug_image) {
                    img.src = data.debug_image;
                    img.style.display = "inline-block";
                } else {
                    img.style.display = "none";
                }
                modal.style.display = "flex";

                // Setup import logic for when user clicks "Import to Canvas"
                const btnImport = document.getElementById("btnAiDebugImport");
                // Remove any old listeners by cloning (simple way)
                const newBtnImport = btnImport.cloneNode(true);
                btnImport.parentNode.replaceChild(newBtnImport, btnImport);

                newBtnImport.addEventListener("click", () => {
                    modal.style.display = "none";
                    document.getElementById("statusText").innerText = `✅ Imported ${data.components.length} components.`;

                    // Dynamically calculate SCALE_FACTOR based on average component width
                    // This prevents components from being placed too far apart on high-res images
                    let sumW = 0, countW = 0;
                    data.components.forEach(c => {
                        const NON_COMP = ['wire', 'junction', 'crossover', 'terminal', 'text'];
                        if (c.box && !NON_COMP.includes(c.type)) {
                            sumW += c.box[2];
                            countW++;
                        }
                    });
                    const avgW = countW > 0 ? (sumW / countW) : 100;
                    // Standard canvas component is ~80px wide. We use 100 for a bit of breathing room.
                    const SCALE_FACTOR = countW > 0 ? (100.0 / avgW) : 1.0;

                    // Extract junction centers for wire merging
                    // Junctions are NOT components — they are wire connection points
                    const NON_COMPONENT_TYPES = ['wire', 'junction', 'crossover', 'terminal', 'text'];
                    const junctionPoints = data.components
                        .filter(c => c.type === 'junction')
                        .map(c => ({
                            x: snap(c.center[0] * SCALE_FACTOR),
                            y: snap(c.center[1] * SCALE_FACTOR)
                        }));

                    components = data.components
                        .filter(c => !NON_COMPONENT_TYPES.includes(c.type))
                        .map(c => {
                            const type = c.type;
                            const db = COMPONENT_DB[type];
                            const params = db ? Object.assign({}, db.params) : { value: '1k' };

                            // Override value from OCR if available
                            if (c.value && c.value !== "TEXT_FOUND") {
                                if (params.value !== undefined) params.value = c.value;
                                else if (params.dc !== undefined) params.dc = c.value;
                            }

                            return {
                                type,
                                name: c.name,
                                value: c.value === "TEXT_FOUND" ? (params.value || '') : (c.value || ''),
                                x: snap(c.center[0] * SCALE_FACTOR),
                                y: snap(c.center[1] * SCALE_FACTOR),
                                params,
                                nodes: c.nodes || []
                            };
                        });

                    // Rebuild name counters from imported data
                    nameCounts = {};
                    components.forEach(c => {
                        const db = COMPONENT_DB[c.type];
                        if (db) {
                            const match = c.name.match(/\d+$/);
                            if (match) {
                                const num = parseInt(match[0]);
                                nameCounts[db.prefix] = Math.max(nameCounts[db.prefix] || 0, num);
                            }
                        }
                    });

                    // Load Wires (Logical Pin-to-Pin connection)
                    // Map original component index to filtered components index
                    const compIdxMap = {};
                    let filteredIdx = 0;
                    data.components.forEach((c, idx) => {
                        if (!NON_COMPONENT_TYPES.includes(c.type)) {
                            compIdxMap[idx] = filteredIdx++;
                        } else {
                            compIdxMap[idx] = -1;
                        }
                    });

                    // Load Wires (Logical Pin-to-Pin connection)
                    wires = [];
                    if (data.connections) {
                        data.connections.forEach(conn => {
                            const idx1 = compIdxMap[conn.pin1.comp_idx];
                            const idx2 = compIdxMap[conn.pin2.comp_idx];
                            
                            if (idx1 !== undefined && idx1 !== -1 && idx2 !== undefined && idx2 !== -1) {
                                const comp1 = components[idx1];
                                const comp2 = components[idx2];
                                
                                const pins1 = getCompPins(comp1);
                                const pins2 = getCompPins(comp2);
                                
                                const p1 = pins1[conn.pin1.pin_id];
                                const p2 = pins2[conn.pin2.pin_id];
                                
                                if (p1 && p2 && (p1.x !== p2.x || p1.y !== p2.y)) {
                                    // Try A* routing first
                                    const astarSegments = routeAStar(p1, p2, components, gridSize);
                                    if (astarSegments && astarSegments.length > 0) {
                                        wires.push(...astarSegments);
                                    } else {
                                        // Fallback to naive Manhattan routing if A* blocked
                                        if (p1.x === p2.x || p1.y === p2.y) {
                                            wires.push([{ x: p1.x, y: p1.y }, { x: p2.x, y: p2.y }]);
                                        } else {
                                            const mid = { x: p2.x, y: p1.y };
                                            wires.push([{ x: p1.x, y: p1.y }, { x: mid.x, y: mid.y }]);
                                            wires.push([{ x: mid.x, y: mid.y }, { x: p2.x, y: p2.y }]);
                                        }
                                    }
                                }
                            }
                        });
                    }

                    // Auto-Center and Zoom Camera dynamically to fit the imported circuit
                    if (components.length > 0 || wires.length > 0) {
                        let minX = Infinity, maxX = -Infinity;
                        let minY = Infinity, maxY = -Infinity;

                        components.forEach(c => {
                            minX = Math.min(minX, c.x);
                            maxX = Math.max(maxX, c.x);
                            minY = Math.min(minY, c.y);
                            maxY = Math.max(maxY, c.y);
                        });

                        wires.forEach(w => {
                            w.forEach(pt => {
                                minX = Math.min(minX, pt.x);
                                maxX = Math.max(maxX, pt.x);
                                minY = Math.min(minY, pt.y);
                                maxY = Math.max(maxY, pt.y);
                            });
                        });

                        // Add a margin
                        const margin = 80;
                        const circuitWidth = (maxX - minX) || 100;
                        const circuitHeight = (maxY - minY) || 100;

                        const canvasW = canvas.width || 800;
                        const canvasH = canvas.height || 600;

                        // Compute optimal zoom to fit all components/wires within the canvas
                        const zoomX = (canvasW - margin * 2) / circuitWidth;
                        const zoomY = (canvasH - margin * 2) / circuitHeight;
                        zoom = Math.min(zoomX, zoomY);

                        // Clamp zoom to reasonable levels (0.2 to 2.0)
                        zoom = Math.max(0.2, Math.min(2.0, zoom));

                        // Center the circuit on the canvas
                        const circuitCenterX = (minX + maxX) / 2;
                        const circuitCenterY = (minY + maxY) / 2;
                        offsetX = canvasW / 2 - circuitCenterX * zoom;
                        offsetY = canvasH / 2 - circuitCenterY * zoom;
                    } else {
                        offsetX = 100;
                        offsetY = 100;
                        zoom = 1.0;
                    }
                    selectedComp = null;
                    updatePropertiesPanel();
                    render();
                });
            }
        } catch (err) {
            console.error(err);
            document.getElementById("statusText").innerText = "❌ AI import failed.";
        }
    });

    // Close logic for the AI Debug modal
    const closeAiDebugModal = () => {
        document.getElementById("aiDebugModal").style.display = "none";
        document.getElementById("statusText").innerText = "❌ Import cancelled.";
    };
    document.getElementById("aiDebugModalClose").addEventListener("click", closeAiDebugModal);
    document.getElementById("btnAiDebugCancel").addEventListener("click", closeAiDebugModal);

    // ═══════════════════════════════════════════
    // INITIAL DRAW
    // ═══════════════════════════════════════════
    render();
});
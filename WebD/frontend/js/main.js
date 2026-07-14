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
            prefix: 'X', label: 'Op-Amp',
            params: { model: 'LM741', vs_pos: '15', vs_neg: '-15' },
            spice: '',  // handled server-side via subcircuit instantiation
            pins: [[-30, -20], [-30, 20], [30, 0]],  // V+, V−, Out
            hitbox: { w: 70, h: 60 }
        },
        ic: {
            prefix: 'X', label: 'IC',
            params: { subckt_name: 'MyIC', num_pins: '2', custom_subckt: '' },
            spice: '',  // handled server-side via X-prefix subcircuit instantiation
            pins: [[-40, 0], [40, 0]],
            hitbox: { w: 80, h: 50 }
        },
        transformer: {
            prefix: 'T', label: 'Transformer',
            params: { value: '1m', coupling: '0.99' },
            spice: '',  // handled server-side as coupled inductors + K line
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
    let placementRotation = 0;

    let undoStack = [];
    let redoStack = [];
    let hasSavedStateForThisDrag = false;

    let currentThemeColors = {
        canvasBg: "#090d16",
        gridColor: "rgba(255, 255, 255, 0.08)",
        wireColor: "#4fc1ff",
        wireSelected: "#38bdf8",
        componentColor: "#cbd5e1",
        labelColor: "#94a3b8",
        valueColor: "#f97316"
    };

    // Initialize Theme
    const savedTheme = localStorage.getItem("theme") || "dark";
    document.documentElement.setAttribute("data-theme", savedTheme);

    const themeBtn = document.getElementById("btnThemeToggle");
    if (themeBtn) {
        themeBtn.addEventListener("click", () => {
            const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
            const newTheme = currentTheme === "dark" ? "light" : "dark";
            document.documentElement.setAttribute("data-theme", newTheme);
            localStorage.setItem("theme", newTheme);
            render();
        });
    }

    function saveState() {
        const state = {
            components: JSON.parse(JSON.stringify(components)),
            wires: JSON.parse(JSON.stringify(wires))
        };
        undoStack.push(state);
        if (undoStack.length > 50) {
            undoStack.shift();
        }
        redoStack = [];
        updateUndoRedoButtons();
    }

    function undo() {
        if (undoStack.length === 0) return;
        const currentState = {
            components: JSON.parse(JSON.stringify(components)),
            wires: JSON.parse(JSON.stringify(wires))
        };
        redoStack.push(currentState);
        
        const previousState = undoStack.pop();
        components = previousState.components;
        wires = previousState.wires;
        
        selectedComponents = [];
        selectedComp = null;
        selectedWires = [];
        selectedWirePts = [];
        
        updateUndoRedoButtons();
        updatePropertiesPanel();
        render();
    }

    function redo() {
        if (redoStack.length === 0) return;
        const currentState = {
            components: JSON.parse(JSON.stringify(components)),
            wires: JSON.parse(JSON.stringify(wires))
        };
        undoStack.push(currentState);
        
        const nextState = redoStack.pop();
        components = nextState.components;
        wires = nextState.wires;
        
        selectedComponents = [];
        selectedComp = null;
        selectedWires = [];
        selectedWirePts = [];
        
        updateUndoRedoButtons();
        updatePropertiesPanel();
        render();
    }

    function updateUndoRedoButtons() {
        const undoBtn = document.getElementById('btn-undo');
        const redoBtn = document.getElementById('btn-redo');
        if (undoBtn) {
            undoBtn.disabled = (undoStack.length === 0);
        }
        if (redoBtn) {
            redoBtn.disabled = (redoStack.length === 0);
        }
    }

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
    
    // Console and Highlighting State
    let nodeMap = {};
    let highlightedComponents = [];
    let highlightedNodes = [];
    let highlightedNetlistLine = null;
    let consoleLogs = [];
    let consoleFilter = 'all';
    let lastNetlistText = "";
    let isManualNetlist = false;

    // Camera & Viewport
    let zoom = 1.0;
    let offsetX = 0;
    let offsetY = 0;
    let isPanning = false;
    let panStart = { x: 0, y: 0 };

    // Interaction State
    let selectedComponents = []; // Array of all selected components
    let selectedComp = null;      // Primary selected component
    let selectedWires = [];      // Selected wire segments
    let selectedWirePts = [];     // Array of selected wire endpoint coordinates {x, y}
    let isDragging = false;
    let dragStart = { x: 0, y: 0 };
    let wireStart = null;
    let mousePos = { x: 0, y: 0 };
    let attachedWireEndpoints = []; // Attached rubber-banding endpoints
    // Pre-existing collision pairs captured at drag-start so that overlapping
    // components can be selected and moved OUT of a collision (not blocked by it).
    let preExistingOverlapPairs = []; // Array of [compA, compB] already overlapping before drag
    let preExistingWireOverlaps = []; // Array of [comp, wire] already overlapping before drag

    // Selection box state
    let selectionStart = null;
    let selectionEnd = null;
    let isSelectingBox = false;

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
        const rotRad = ((comp.rotation || 0) % 360) * Math.PI / 180;
        const cosR = Math.round(Math.cos(rotRad));
        const sinR = Math.round(Math.sin(rotRad));
        return db.pins.map(([dx, dy]) => {
            const rx = dx * cosR - dy * sinR;
            const ry = dx * sinR + dy * cosR;
            return {
                x: comp.x + rx,
                y: comp.y + ry
            };
        });
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
                const p2 = wire[i + 1];
                const l2 = (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2;
                if (l2 === 0) continue;

                // Point to line segment projection
                let t = ((rawWorldX - p1.x) * (p2.x - p1.x) + (rawWorldY - p1.y) * (p2.y - p1.y)) / l2;
                t = Math.max(0, Math.min(1, t));
                const projX = p1.x + t * (p2.x - p1.x);
                const projY = p1.y + t * (p2.y - p1.y);

                const dist = Math.sqrt((rawWorldX - projX) ** 2 + (rawWorldY - projY) ** 2);
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
                value: '1k', params: { value: '1k' },
                rotation: 0
            };
        }
        const name = nextName(db.prefix);
        const params = Object.assign({}, db.params);
        return {
            type, x: worldX, y: worldY,
            name,
            value: params.value || params.dc || params.mag || '',
            params,
            rotation: 0
        };
    }

    // Helper to check if a horizontal or vertical segment between two points intersects
    // any component's bounding box (hitbox), excluding a list of components.
    function doesSegmentIntersectComponent(pA, pB, allComponents, excludeComps = []) {
        const IGNORE_ROUTING = new Set(['label', 'junction', 'crossover', 'terminal']);
        for (const c of allComponents) {
            if (excludeComps.includes(c) || IGNORE_ROUTING.has(c.type)) continue;
            const db = COMPONENT_DB[c.type];
            const hb = db ? db.hitbox : { w: 40, h: 40 };
            const rot = c.rotation || 0;
            const w = (rot === 90 || rot === 270) ? hb.h : hb.w;
            const h = (rot === 90 || rot === 270) ? hb.w : hb.h;
            
            // Shrink the hitbox slightly (e.g. by 2 pixels) to avoid snapping false positives on the exact border
            const margin = 2;
            const xMin = c.x - w / 2 + margin;
            const xMax = c.x + w / 2 - margin;
            const yMin = c.y - h / 2 + margin;
            const yMax = c.y + h / 2 - margin;
            
            if (Math.abs(pA.y - pB.y) < 1) {
                // Horizontal segment
                const y = pA.y;
                const minX = Math.min(pA.x, pB.x);
                const maxX = Math.max(pA.x, pB.x);
                if (y > yMin && y < yMax && minX < xMax && maxX > xMin) {
                    return true;
                }
            } else if (Math.abs(pA.x - pB.x) < 1) {
                // Vertical segment
                const x = pA.x;
                const minY = Math.min(pA.y, pB.y);
                const maxY = Math.max(pA.y, pB.y);
                if (x > xMin && x < xMax && minY < yMax && maxY > yMin) {
                    return true;
                }
            }
        }
        return false;
    }

    function componentsOverlap(c1, c2) {
        const db1 = COMPONENT_DB[c1.type];
        const hb1 = db1 ? db1.hitbox : { w: 40, h: 40 };
        const rot1 = c1.rotation || 0;
        const w1 = (rot1 === 90 || rot1 === 270) ? hb1.h : hb1.w;
        const h1 = (rot1 === 90 || rot1 === 270) ? hb1.w : hb1.h;

        const db2 = COMPONENT_DB[c2.type];
        const hb2 = db2 ? db2.hitbox : { w: 40, h: 40 };
        const rot2 = c2.rotation || 0;
        const w2 = (rot2 === 90 || rot2 === 270) ? hb2.h : hb2.w;
        const h2 = (rot2 === 90 || rot2 === 270) ? hb2.w : hb2.h;

        const rect1 = {
            left: c1.x - w1 / 2,
            right: c1.x + w1 / 2,
            top: c1.y - h1 / 2,
            bottom: c1.y + h1 / 2
        };

        const rect2 = {
            left: c2.x - w2 / 2,
            right: c2.x + w2 / 2,
            top: c2.y - h2 / 2,
            bottom: c2.y + h2 / 2
        };

        return !(rect1.right <= rect2.left ||
                 rect1.left >= rect2.right ||
                 rect1.bottom <= rect2.top ||
                 rect1.top >= rect2.bottom);
    }

    function isOverlappingAny(comp, allComponents, excludeList = []) {
        if (comp.type === 'label') return false;
        for (const c of allComponents) {
            if (c === comp || excludeList.includes(c) || c.type === 'label') continue;
            if (componentsOverlap(comp, c)) {
                return true;
            }
        }
        return false;
    }

    function isWireSegmentGoingInwards(c, pinPos, Q) {
        const db = COMPONENT_DB[c.type];
        if (!db || !db.pins) return false;
        
        const rot = c.rotation || 0;
        let matchedPinIdx = -1;
        
        const pins = getCompPins(c);
        for (let i = 0; i < pins.length; i++) {
            if (Math.abs(pins[i].x - pinPos.x) < 1 && Math.abs(pins[i].y - pinPos.y) < 1) {
                matchedPinIdx = i;
                break;
            }
        }
        if (matchedPinIdx === -1) return false;
        
        const localPin = db.pins[matchedPinIdx];
        const rx = localPin[0];
        const ry = localPin[1];
        
        let rotRx = rx;
        let rotRy = ry;
        if (rot === 90) {
            rotRx = -ry;
            rotRy = rx;
        } else if (rot === 180) {
            rotRx = -rx;
            rotRy = -ry;
        } else if (rot === 270) {
            rotRx = ry;
            rotRy = -rx;
        }
        
        const isHorizontal = Math.abs(pinPos.y - Q.y) < 1;
        const isVertical = Math.abs(pinPos.x - Q.x) < 1;
        
        if (isHorizontal) {
            if (rotRx > 0 && Q.x < pinPos.x) return true;
            if (rotRx < 0 && Q.x > pinPos.x) return true;
        }
        if (isVertical) {
            if (rotRy > 0 && Q.y < pinPos.y) return true;
            if (rotRy < 0 && Q.y > pinPos.y) return true;
        }
        return false;
    }

    function doesCompOverlapAnyWire(comp, allWires) {
        // Labels, junctions, and crossovers are intentionally placed ON wires —
        // they must never be blocked by the wire-overlap guard.
        const WIRE_TRANSPARENT = new Set(['label', 'junction', 'crossover', 'terminal']);
        if (WIRE_TRANSPARENT.has(comp.type)) return false;

        const pins = getCompPins(comp);
        for (const wire of allWires) {
            if (wire.length < 2) continue;
            const p1 = wire[0];
            const p2 = wire[1];
            
            const hasP1 = pins.some(p => Math.abs(p.x - p1.x) < 1 && Math.abs(p.y - p1.y) < 1);
            const hasP2 = pins.some(p => Math.abs(p.x - p2.x) < 1 && Math.abs(p.y - p2.y) < 1);
            
            if (hasP1) {
                if (isWireSegmentGoingInwards(comp, p1, p2)) return true;
                continue;
            }
            if (hasP2) {
                if (isWireSegmentGoingInwards(comp, p2, p1)) return true;
                continue;
            }
            
            if (doesSegmentIntersectComponent(p1, p2, [comp])) {
                return true;
            }
        }
        return false;
    }


    // Route a straight horizontal or vertical segment around any component it intersects
    function routeAroundComponent(pA, pB, allComponents, excludeComps = []) {
        const IGNORE_ROUTING = new Set(['label', 'junction', 'crossover', 'terminal']);
        const offset = (typeof gridSize !== 'undefined') ? gridSize : 20;
        for (const c of allComponents) {
            if (excludeComps.includes(c) || IGNORE_ROUTING.has(c.type)) continue;
            const db = COMPONENT_DB[c.type];
            const hb = db ? db.hitbox : { w: 40, h: 40 };
            const rot = c.rotation || 0;
            const w = (rot === 90 || rot === 270) ? hb.h : hb.w;
            const h = (rot === 90 || rot === 270) ? hb.w : hb.h;
            
            const margin = 2;
            const xMin = snap(c.x - w / 2 - offset);
            const xMax = snap(c.x + w / 2 + offset);
            const yMin = snap(c.y - h / 2 - offset);
            const yMax = snap(c.y + h / 2 + offset);
            
            if (Math.abs(pA.y - pB.y) < 1) {
                // Horizontal segment
                const y = pA.y;
                const minX = Math.min(pA.x, pB.x);
                const maxX = Math.max(pA.x, pB.x);
                
                const cXMin = c.x - w / 2 + margin;
                const cXMax = c.x + w / 2 - margin;
                const cYMin = c.y - h / 2 + margin;
                const cYMax = c.y + h / 2 - margin;
                
                if (y > cYMin && y < cYMax && minX < cXMax && maxX > cXMin) {
                    const detourY = (y - cYMin < cYMax - y) ? yMin : yMax;
                    
                    const pt1 = { x: pA.x, y: pA.y };
                    const pt2 = { x: snap(pA.x < pB.x ? xMin : xMax), y: pA.y };
                    const pt3 = { x: pt2.x, y: snap(detourY) };
                    const pt4 = { x: snap(pA.x < pB.x ? xMax : xMin), y: pt3.y };
                    const pt5 = { x: pt4.x, y: pB.y };
                    const pt6 = { x: pB.x, y: pB.y };
                    
                    return [
                        [pt1, pt2],
                        [pt2, pt3],
                        [pt3, pt4],
                        [pt4, pt5],
                        [pt5, pt6]
                    ];
                }
            } else if (Math.abs(pA.x - pB.x) < 1) {
                // Vertical segment
                const x = pA.x;
                const minY = Math.min(pA.y, pB.y);
                const maxY = Math.max(pA.y, pB.y);
                
                const cXMin = c.x - w / 2 + margin;
                const cXMax = c.x + w / 2 - margin;
                const cYMin = c.y - h / 2 + margin;
                const cYMax = c.y + h / 2 - margin;
                
                if (x > cXMin && x < cXMax && minY < cYMax && maxY > cYMin) {
                    const detourX = (x - cXMin < cXMax - x) ? xMin : xMax;
                    
                    const pt1 = { x: pA.x, y: pA.y };
                    const pt2 = { x: pA.x, y: snap(pA.y < pB.y ? yMin : yMax) };
                    const pt3 = { x: snap(detourX), y: pt2.y };
                    const pt4 = { x: pt3.x, y: snap(pA.y < pB.y ? yMax : yMin) };
                    const pt5 = { x: pB.x, y: pt4.y };
                    const pt6 = { x: pB.x, y: pB.y };
                    
                    return [
                        [pt1, pt2],
                        [pt2, pt3],
                        [pt3, pt4],
                        [pt4, pt5],
                        [pt5, pt6]
                    ];
                }
            }
        }
        return [[pA, pB]];
    }

    // Reroute all wires in the canvas outward from the components' center of mass
    function rerouteAllWires() {
        // Clean up redundant junctions (degree < 3 in the current wires list)
        const junctionDegrees = {};
        wires.forEach(w => {
            if (w.length >= 2) {
                const k1 = Math.round(w[0].x) + "," + Math.round(w[0].y);
                const k2 = Math.round(w[1].x) + "," + Math.round(w[1].y);
                junctionDegrees[k1] = (junctionDegrees[k1] || 0) + 1;
                junctionDegrees[k2] = (junctionDegrees[k2] || 0) + 1;
            }
        });
        components = components.filter(c => {
            if (c.type === 'junction') {
                const key = Math.round(c.x) + "," + Math.round(c.y);
                const deg = junctionDegrees[key] || 0;
                return deg >= 3;
            }
            return true;
        });

        if (components.length === 0) return;

        // 1. DSU to find connected points/nodes
        const parent = {};
        function find(i) {
            if (parent[i] === undefined) {
                parent[i] = i;
                return i;
            }
            if (parent[i] === i) return i;
            return parent[i] = find(parent[i]);
        }
        function union(i, j) {
            const rootI = find(i);
            const rootJ = find(j);
            if (rootI !== rootJ) {
                parent[rootI] = rootJ;
            }
        }

        const gSize = gridSize || 20;
        function getInterpolatedPoints(p1, p2) {
            const ax = Math.round(p1.x / gSize) * gSize;
            const ay = Math.round(p1.y / gSize) * gSize;
            const bx = Math.round(p2.x / gSize) * gSize;
            const by = Math.round(p2.y / gSize) * gSize;

            const dx = bx - ax;
            const dy = by - ay;

            const stepsX = dx !== 0 ? Math.floor(Math.abs(dx) / gSize) : 0;
            const stepsY = dy !== 0 ? Math.floor(Math.abs(dy) / gSize) : 0;
            const steps = Math.max(stepsX, stepsY);

            const pts = [];
            pts.push({ x: ax, y: ay });
            if (steps > 0) {
                for (let i = 1; i <= steps; i++) {
                    const t = i / steps;
                    const ix = Math.round((ax + dx * t) / gSize) * gSize;
                    const iy = Math.round((ay + dy * t) / gSize) * gSize;
                    pts.push({ x: ix, y: iy });
                }
            }
            return pts;
        }

        // Count degrees of all coordinate steps
        const coordDegree = {};

        // Union endpoints and intermediate points of all current wires
        wires.forEach(w => {
            if (w.length >= 2) {
                const pts = getInterpolatedPoints(w[0], w[1]);
                for (let i = 0; i < pts.length - 1; i++) {
                    const k1 = pts[i].x + "," + pts[i].y;
                    const k2 = pts[i + 1].x + "," + pts[i + 1].y;
                    union(k1, k2);
                    
                    coordDegree[k1] = (coordDegree[k1] || 0) + 1;
                    coordDegree[k2] = (coordDegree[k2] || 0) + 1;
                }
            }
        });

        // Group component pins by DSU root
        const netGroups = {};
        components.forEach((c, compIdx) => {
            const pins = getCompPins(c);
            pins.forEach((p, pinIdx) => {
                const key = Math.round(p.x) + "," + Math.round(p.y);
                const root = find(key);
                if (!netGroups[root]) {
                    netGroups[root] = [];
                }
                netGroups[root].push({ compIdx, pinIdx, x: p.x, y: p.y });
            });
        });

        // Add virtual pins for junction points (degree >= 3) to prevent them from disappearing during rerouting
        for (const [key, degree] of Object.entries(coordDegree)) {
            if (degree >= 3) {
                const [xStr, yStr] = key.split(",");
                const jx = parseInt(xStr);
                const jy = parseInt(yStr);
                const root = find(key);
                if (!netGroups[root]) {
                    netGroups[root] = [];
                }
                const alreadyExists = netGroups[root].some(p => Math.abs(p.x - jx) < 1 && Math.abs(p.y - jy) < 1);
                if (!alreadyExists) {
                    netGroups[root].push({ compIdx: -1, pinIdx: -1, x: jx, y: jy });
                }
            }
        }

        // 2. Select minimal optimal pin-to-pin connections via Kruskal's MST
        const connectionsToRoute = [];
        Object.keys(netGroups).forEach(root => {
            const groupPins = netGroups[root];
            if (groupPins.length < 2) return;

            const edges = [];
            for (let i = 0; i < groupPins.length; i++) {
                for (let j = i + 1; j < groupPins.length; j++) {
                    const pA = groupPins[i];
                    const pB = groupPins[j];
                    const dist = Math.abs(pA.x - pB.x) + Math.abs(pA.y - pB.y);
                    edges.push({ i, j, dist, pA, pB });
                }
            }
            edges.sort((a, b) => a.dist - b.dist);

            const groupParent = {};
            function gFind(i) {
                if (groupParent[i] === undefined) {
                    groupParent[i] = i;
                    return i;
                }
                if (groupParent[i] === i) return i;
                return groupParent[i] = gFind(groupParent[i]);
            }
            function gUnion(i, j) {
                const rI = gFind(i);
                const rJ = gFind(j);
                if (rI !== rJ) {
                    groupParent[rI] = rJ;
                    return true;
                }
                return false;
            }

            edges.forEach(edge => {
                if (gUnion(edge.i, edge.j)) {
                    connectionsToRoute.push({ p1: edge.pA, p2: edge.pB });
                }
            });
        });

        // Calculate center of mass of all components
        let sumX = 0, sumY = 0;
        components.forEach(c => {
            sumX += c.x;
            sumY += c.y;
        });
        const Cx = sumX / components.length;
        const Cy = sumY / components.length;

        // 3. Destroy all existing wires
        wires = [];

        // 4. Connect again outward from center with collision-aware routing
        function getExcludeListForSegment(pA, pB, comp1Idx, comp2Idx) {
            const exclude = [];
            if (comp1Idx !== undefined && comp1Idx !== -1) {
                const c1 = components[comp1Idx];
                if (c1 && !isWireSegmentGoingInwards(c1, pA, pB)) {
                    exclude.push(c1);
                }
            }
            if (comp2Idx !== undefined && comp2Idx !== -1) {
                const c2 = components[comp2Idx];
                if (c2 && !isWireSegmentGoingInwards(c2, pB, pA)) {
                    exclude.push(c2);
                }
            }
            return exclude;
        }

        connectionsToRoute.forEach(conn => {
            const p1 = conn.p1;
            const p2 = conn.p2;

            if (p1.x !== p2.x && p1.y !== p2.y) {
                const mid1 = { x: p2.x, y: p1.y }; // H-then-V
                const mid2 = { x: p1.x, y: p2.y }; // V-then-H

                // Build exclude lists for collision checks
                const excl_p1_mid1 = getExcludeListForSegment(p1, mid1, p1.compIdx, -1);
                const excl_mid1_p2 = getExcludeListForSegment(mid1, p2, -1, p2.compIdx);

                const excl_p1_mid2 = getExcludeListForSegment(p1, mid2, p1.compIdx, -1);
                const excl_mid2_p2 = getExcludeListForSegment(mid2, p2, -1, p2.compIdx);

                // Avoid routing paths that enter the connected components inwards through their bodies
                const inwards1 = 
                    (p1.compIdx !== -1 && isWireSegmentGoingInwards(components[p1.compIdx], p1, mid1)) ||
                    (p2.compIdx !== -1 && isWireSegmentGoingInwards(components[p2.compIdx], p2, mid1));
                const inwards2 = 
                    (p1.compIdx !== -1 && isWireSegmentGoingInwards(components[p1.compIdx], p1, mid2)) ||
                    (p2.compIdx !== -1 && isWireSegmentGoingInwards(components[p2.compIdx], p2, mid2));

                const coll1 = inwards1 || 
                              doesSegmentIntersectComponent(p1, mid1, components, excl_p1_mid1) || 
                              doesSegmentIntersectComponent(mid1, p2, components, excl_mid1_p2);
                const coll2 = inwards2 || 
                              doesSegmentIntersectComponent(p1, mid2, components, excl_p1_mid2) || 
                              doesSegmentIntersectComponent(mid2, p2, components, excl_mid2_p2);

                const dist1 = Math.pow(mid1.x - Cx, 2) + Math.pow(mid1.y - Cy, 2);
                const dist2 = Math.pow(mid2.x - Cx, 2) + Math.pow(mid2.y - Cy, 2);

                let chosenSegments = [];
                let chosenExcludes = [];
                if (coll1 && !coll2) {
                    chosenSegments = [[p1, mid2], [mid2, p2]];
                    chosenExcludes = [excl_p1_mid2, excl_mid2_p2];
                } else if (coll2 && !coll1) {
                    chosenSegments = [[p1, mid1], [mid1, p2]];
                    chosenExcludes = [excl_p1_mid1, excl_mid1_p2];
                } else {
                    if (dist1 > dist2) {
                        chosenSegments = [[p1, mid1], [mid1, p2]];
                        chosenExcludes = [excl_p1_mid1, excl_mid1_p2];
                    } else {
                        chosenSegments = [[p1, mid2], [mid2, p2]];
                        chosenExcludes = [excl_p1_mid2, excl_mid2_p2];
                    }
                }

                chosenSegments.forEach((seg, idx) => {
                    if (seg[0].x !== seg[1].x || seg[0].y !== seg[1].y) {
                        const routed = routeAroundComponent(seg[0], seg[1], components, chosenExcludes[idx]);
                        routed.forEach(rSeg => wires.push(rSeg));
                    }
                });
            } else {
                const excl_straight = getExcludeListForSegment(p1, p2, p1.compIdx, p2.compIdx);
                const routed = routeAroundComponent(p1, p2, components, excl_straight);
                routed.forEach(rSeg => wires.push(rSeg));
            }
        });

        // 5. Merge contiguous collinear segments to clean up
        let merged = true;
        while (merged) {
            merged = false;
            for (let i = 0; i < wires.length; i++) {
                for (let j = i + 1; j < wires.length; j++) {
                    const w1 = wires[i];
                    const w2 = wires[j];

                    let shared = null;
                    let other1 = null;
                    let other2 = null;

                    const ptsMatch = (a, b) => Math.abs(a.x - b.x) < 1 && Math.abs(a.y - b.y) < 1;

                    if (ptsMatch(w1[0], w2[0])) { shared = w1[0]; other1 = w1[1]; other2 = w2[1]; }
                    else if (ptsMatch(w1[0], w2[1])) { shared = w1[0]; other1 = w1[1]; other2 = w2[0]; }
                    else if (ptsMatch(w1[1], w2[0])) { shared = w1[1]; other1 = w1[0]; other2 = w2[1]; }
                    else if (ptsMatch(w1[1], w2[1])) { shared = w1[1]; other1 = w1[0]; other2 = w2[0]; }

                    if (shared) {
                        const isH1 = Math.abs(w1[0].y - w1[1].y) < 1;
                        const isV1 = Math.abs(w1[0].x - w1[1].x) < 1;
                        const isH2 = Math.abs(w2[0].y - w2[1].y) < 1;
                        const isV2 = Math.abs(w2[0].x - w2[1].x) < 1;

                        if ((isH1 && isH2 && Math.abs(other1.y - other2.y) < 1) || 
                            (isV1 && isV2 && Math.abs(other1.x - other2.x) < 1)) {
                            wires[i] = [other1, other2];
                            wires.splice(j, 1);
                            merged = true;
                            break;
                        }
                    }
                }
                if (merged) break;
            }
        }

        // Filter out zero-length segments
        wires = wires.filter(w => !(Math.abs(w[0].x - w[1].x) < 1 && Math.abs(w[0].y - w[1].y) < 1));

        render();
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
            // Clear interactive debug highlights on click
            if (highlightedComponents.length > 0 || highlightedNodes.length > 0 || highlightedNetlistLine !== null) {
                highlightedComponents = [];
                highlightedNodes = [];
                highlightedNetlistLine = null;
                scheduleAnimation();
                updateNetlistPreview(lastNetlistText);
                render();
            }

            if (mode === 'select') {
                const worldPos = screenToWorld(e.offsetX, e.offsetY);
                const rawWorld = screenToWorldRaw(e.offsetX, e.offsetY);
                const hit = hitTest(worldPos);

                // Check if we hit a wire endpoint/junction (within radius)
                let hitWirePt = null;
                for (const wire of wires) {
                    for (const pt of wire) {
                        const dx = rawWorld.x - pt.x;
                        const dy = rawWorld.y - pt.y;
                        if (Math.sqrt(dx * dx + dy * dy) < PIN_SNAP_RADIUS) {
                            hitWirePt = pt;
                            break;
                        }
                    }
                    if (hitWirePt) break;
                }

                let hitWire = null;
                if (!hit && !hitWirePt) {
                    hitWire = hitTestWire(rawWorld);
                }

                if (hit) {
                    // Clicking on a component
                    if (!e.shiftKey && !e.ctrlKey) {
                        selectedWires = [];
                    }
                    if (e.shiftKey || e.ctrlKey) {
                        if (selectedComponents.includes(hit)) {
                            selectedComponents = selectedComponents.filter(c => c !== hit);
                            selectedComp = selectedComponents[selectedComponents.length - 1] || null;
                        } else {
                            selectedComponents.push(hit);
                            selectedComp = hit;
                        }
                    } else {
                        if (!selectedComponents.includes(hit)) {
                            selectedComponents = [hit];
                            selectedComp = hit;
                            selectedWirePts = [];
                        }
                    }

                    isDragging = true;
                    dragStart = { x: worldPos.x, y: worldPos.y };

                    // Snapshot pre-existing overlaps so the user can drag
                    // overlapping components OUT of a collision freely.
                    preExistingOverlapPairs = [];
                    preExistingWireOverlaps = [];
                    selectedComponents.forEach(sel => {
                        // Component-component pre-existing pairs
                        components.forEach(other => {
                            if (other === sel || selectedComponents.includes(other)) return;
                            if (componentsOverlap(sel, other)) {
                                preExistingOverlapPairs.push([sel, other]);
                            }
                        });
                        // Component-wire pre-existing pairs
                        wires.forEach(wire => {
                            if (doesCompOverlapAnyWire(sel, [wire])) {
                                preExistingWireOverlaps.push([sel, wire]);
                            }
                        });
                    });

                    // Populate attached rubber-banding wire endpoints for all selected components
                    attachedWireEndpoints = [];
                    selectedComponents.forEach(comp => {
                        const pins = getCompPins(comp);
                        wires.forEach(wire => {
                            if (wire.length < 2) return;
                            [wire[0], wire[wire.length - 1]].forEach(pt => {
                                // Skip if the wire point is already in selectedWirePts (moved directly)
                                if (selectedWirePts.includes(pt)) return;

                                const pinIdx = pins.findIndex(pin => Math.abs(pin.x - pt.x) < 1 && Math.abs(pin.y - pt.y) < 1);
                                if (pinIdx !== -1) {
                                    if (!attachedWireEndpoints.some(att => att.pt === pt)) {
                                        attachedWireEndpoints.push({ pt, comp, pinIdx });
                                    }
                                }
                            });
                        });
                    });

                    updatePropertiesPanel();
                } else if (hitWirePt) {
                    // Clicking on a wire endpoint/junction
                    if (!e.shiftKey && !e.ctrlKey) {
                        selectedWires = [];
                    }
                    if (e.shiftKey || e.ctrlKey) {
                        if (selectedWirePts.includes(hitWirePt)) {
                            selectedWirePts = selectedWirePts.filter(pt => pt !== hitWirePt);
                        } else {
                            selectedWirePts.push(hitWirePt);
                        }
                    } else {
                        if (!selectedWirePts.includes(hitWirePt)) {
                            selectedWirePts = [hitWirePt];
                            selectedComponents = [];
                            selectedComp = null;
                        }
                    }

                    isDragging = true;
                    dragStart = { x: worldPos.x, y: worldPos.y };
                    attachedWireEndpoints = [];
                    updatePropertiesPanel();
                } else if (hitWire) {
                    // Clicking on a wire segment
                    if (e.shiftKey || e.ctrlKey) {
                        if (selectedWires.includes(hitWire)) {
                            selectedWires = selectedWires.filter(w => w !== hitWire);
                        } else {
                            selectedWires.push(hitWire);
                        }
                    } else {
                        selectedWires = [hitWire];
                        selectedComponents = [];
                        selectedComp = null;
                        selectedWirePts = [];
                    }
                    updatePropertiesPanel();
                    render();
                } else {
                    // Click on empty space
                    if (!e.shiftKey && !e.ctrlKey) {
                        selectedWires = [];
                        selectedComponents = [];
                        selectedWirePts = [];
                        selectedComp = null;
                        updatePropertiesPanel();
                    }

                    // Start selection box dragging
                    isSelectingBox = true;
                    selectionStart = screenToWorldRaw(e.offsetX, e.offsetY);
                    selectionEnd = { ...selectionStart };
                }
            }
            else if (mode === 'wire') {
                // Use smart snapping for wire endpoints (pin > grid)
                const snapped = smartSnap(e.offsetX, e.offsetY);
                const raw = screenToWorldRaw(e.offsetX, e.offsetY);
                const landedOnPin = findNearestPin(raw.x, raw.y);

                if (!wireStart) {
                    wireStart = snapped;
                    const hitWirePt = findNearestWireSegment(raw.x, raw.y);
                    if (hitWirePt && !landedOnPin) {
                        const alreadyHasJunc = components.some(c => c.type === 'junction' && Math.abs(c.x - snapped.x) < 1 && Math.abs(c.y - snapped.y) < 1);
                        if (!alreadyHasJunc) {
                            components.push(createComponent('junction', snapped.x, snapped.y));
                        }
                    }
                } else {
                    // Manhattan wire routing
                    if (wireStart.x !== snapped.x || wireStart.y !== snapped.y) {
                        saveState();
                        const p1 = { x: wireStart.x, y: wireStart.y };
                        const p2 = { x: snapped.x, y: snapped.y };
                        
                        // Find components to exclude (those connected to either end of the segment)
                        const exclude = [];
                        components.forEach(c => {
                            const pins = getCompPins(c);
                            const hasP1 = pins.some(p => Math.abs(p.x - p1.x) < 1 && Math.abs(p.y - p1.y) < 1);
                            const hasP2 = pins.some(p => Math.abs(p.x - p2.x) < 1 && Math.abs(p.y - p2.y) < 1);
                            if (hasP1 || hasP2) exclude.push(c);
                        });

                        const mid1 = { x: p2.x, y: p1.y };
                        const mid2 = { x: p1.x, y: p2.y };
                        
                        const coll1 = doesSegmentIntersectComponent(p1, mid1, components, exclude) || 
                                      doesSegmentIntersectComponent(mid1, p2, components, exclude);
                        const coll2 = doesSegmentIntersectComponent(p1, mid2, components, exclude) || 
                                      doesSegmentIntersectComponent(mid2, p2, components, exclude);
                        
                        let chosenSegments = [];
                        if (coll1 && !coll2) {
                            // Choose V-then-H
                            chosenSegments = [
                                [p1, { x: p1.x, y: p2.y }],
                                [{ x: p1.x, y: p2.y }, p2]
                            ];
                        } else {
                            // Choose H-then-V
                            chosenSegments = [
                                [p1, { x: p2.x, y: p1.y }],
                                [{ x: p2.x, y: p1.y }, p2]
                            ];
                        }
                        
                        // Route each chosen segment around any obstacle
                        let tempSegments = [];
                        let intersectsComponent = false;
                        chosenSegments.forEach(seg => {
                            if (seg[0].x !== seg[1].x || seg[0].y !== seg[1].y) {
                                const routed = routeAroundComponent(seg[0], seg[1], components, exclude);
                                routed.forEach(rSeg => {
                                    if (doesSegmentIntersectComponent(rSeg[0], rSeg[1], components, exclude)) {
                                        intersectsComponent = true;
                                    }
                                    components.forEach(c => {
                                        const pins = getCompPins(c);
                                        const connectsP1 = pins.some(p => Math.abs(p.x - rSeg[0].x) < 1 && Math.abs(p.y - rSeg[0].y) < 1);
                                        const connectsP2 = pins.some(p => Math.abs(p.x - rSeg[1].x) < 1 && Math.abs(p.y - rSeg[1].y) < 1);
                                        if (connectsP1 && isWireSegmentGoingInwards(c, rSeg[0], rSeg[1])) {
                                            intersectsComponent = true;
                                        }
                                        if (connectsP2 && isWireSegmentGoingInwards(c, rSeg[1], rSeg[0])) {
                                            intersectsComponent = true;
                                        }
                                    });
                                    tempSegments.push(rSeg);
                                });
                            }
                        });

                        if (intersectsComponent) {
                            if (undoStack.length > 0) {
                                undoStack.pop();
                                updateUndoRedoButtons();
                            }
                            document.getElementById("statusText").innerText = "⚠️ Cannot route wire: Blocked by a component.";
                            return;
                        }

                        tempSegments.forEach(rSeg => wires.push(rSeg));
                    }

                    // Place a junction at the wire body snapping point
                    let snappedToWire = false;
                    const hitWirePt = findNearestWireSegment(raw.x, raw.y);
                    if (hitWirePt && !landedOnPin) {
                        const alreadyHasJunc = components.some(c => c.type === 'junction' && Math.abs(c.x - snapped.x) < 1 && Math.abs(c.y - snapped.y) < 1);
                        if (!alreadyHasJunc) {
                            components.push(createComponent('junction', snapped.x, snapped.y));
                        }
                        snappedToWire = true;
                    }

                    // If we landed on a pin or snapped to a wire, auto-terminate the wire
                    if (landedOnPin || snappedToWire) {
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
                comp.rotation = placementRotation || 0;
                if (isOverlappingAny(comp, components) || doesCompOverlapAnyWire(comp, wires)) {
                    document.getElementById("statusText").innerText = "⚠️ Cannot place component: Overlaps another component or wire.";
                    return;
                }
                saveState();
                components.push(comp);
                selectedComponents = [comp];
                selectedComp = comp;
                selectedWirePts = [];
                mode = 'select';
                placementRotation = 0;
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
        else if (isSelectingBox) {
            selectionEnd = screenToWorldRaw(e.offsetX, e.offsetY);
            mousePos = screenToWorld(e.offsetX, e.offsetY);
            hoveredPin = null;
        }
        else if (isDragging && (selectedComponents.length > 0 || selectedWirePts.length > 0)) {
            if (!hasSavedStateForThisDrag) {
                saveState();
                hasSavedStateForThisDrag = true;
            }
            mousePos = screenToWorld(e.offsetX, e.offsetY);
            const dx = mousePos.x - dragStart.x;
            const dy = mousePos.y - dragStart.y;

            if (dx !== 0 || dy !== 0) {
                // Move components
                selectedComponents.forEach(c => {
                    c.x += dx;
                    c.y += dy;
                });

                // Move wire endpoints/junctions
                selectedWirePts.forEach(pt => {
                    pt.x += dx;
                    pt.y += dy;
                });

                // Move attached rubber-banding wires
                attachedWireEndpoints.forEach(att => {
                    att.pt.x += dx;
                    att.pt.y += dy;
                });

                dragStart = { x: mousePos.x, y: mousePos.y };
            }
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
        if (isSelectingBox && selectionStart && selectionEnd) {
            isSelectingBox = false;

            // Bounding box in world coordinates
            const xMin = Math.min(selectionStart.x, selectionEnd.x);
            const xMax = Math.max(selectionStart.x, selectionEnd.x);
            const yMin = Math.min(selectionStart.y, selectionEnd.y);
            const yMax = Math.max(selectionStart.y, selectionEnd.y);

            if (xMax - xMin > 5 || yMax - yMin > 5) {
                if (!e.shiftKey && !e.ctrlKey) {
                    selectedComponents = [];
                    selectedWirePts = [];
                    selectedWires = [];
                }

                // Select components
                components.forEach(c => {
                    if (c.x >= xMin && c.x <= xMax && c.y >= yMin && c.y <= yMax) {
                        if (!selectedComponents.includes(c)) {
                            selectedComponents.push(c);
                        }
                    }
                });

                // Select wires (both endpoints in marquee box)
                wires.forEach(wire => {
                    if (wire.length < 2) return;
                    const p1 = wire[0];
                    const p2 = wire[1];
                    const p1In = p1.x >= xMin && p1.x <= xMax && p1.y >= yMin && p1.y <= yMax;
                    const p2In = p2.x >= xMin && p2.x <= xMax && p2.y >= yMin && p2.y <= yMax;
                    if (p1In && p2In) {
                        if (!selectedWires.includes(wire)) {
                            selectedWires.push(wire);
                        }
                    }
                });

                // Select wire points (junctions/corners)
                wires.forEach(wire => {
                    wire.forEach(pt => {
                        if (pt.x >= xMin && pt.x <= xMax && pt.y >= yMin && pt.y <= yMax) {
                            if (!selectedWirePts.includes(pt)) {
                                selectedWirePts.push(pt);
                            }
                        }
                    });
                });

                selectedComp = selectedComponents[0] || null;
            }

            selectionStart = null;
            selectionEnd = null;
            updatePropertiesPanel();
            render();
        }
        else if (isDragging && (selectedComponents.length > 0 || selectedWirePts.length > 0)) {
            // Snap components to grid
            selectedComponents.forEach(c => {
                c.x = snap(c.x);
                c.y = snap(c.y);
            });

            // Snap selected wire points to grid
            selectedWirePts.forEach(pt => {
                pt.x = snap(pt.x);
                pt.y = snap(pt.y);
            });

            // Snap attached rubber-band wire endpoints precisely to the grid-aligned pins
            attachedWireEndpoints.forEach(att => {
                const snappedPins = getCompPins(att.comp);
                const pin = snappedPins[att.pinIdx];
                att.pt.x = pin.x;
                att.pt.y = pin.y;
            });

            // Auto-orthogonalize any wires that became diagonal during drag (selecting the bend that avoids component collision)
            const newWires = [];
            wires.forEach(wire => {
                if (wire.length === 2) {
                    const p1 = wire[0];
                    const p2 = wire[1];
                    if (p1.x !== p2.x && p1.y !== p2.y) {
                        // Find components to exclude (the ones connected to either end of the segment)
                        const exclude = [];
                        components.forEach(c => {
                            const pins = getCompPins(c);
                            const hasP1 = pins.some(p => Math.abs(p.x - p1.x) < 1 && Math.abs(p.y - p1.y) < 1);
                            const hasP2 = pins.some(p => Math.abs(p.x - p2.x) < 1 && Math.abs(p.y - p2.y) < 1);
                            if (hasP1 || hasP2) exclude.push(c);
                        });

                        const mid1 = { x: p2.x, y: p1.y };
                        const mid2 = { x: p1.x, y: p2.y };
                        
                        const coll1 = doesSegmentIntersectComponent(p1, mid1, components, exclude) || 
                                      doesSegmentIntersectComponent(mid1, p2, components, exclude);
                        const coll2 = doesSegmentIntersectComponent(p1, mid2, components, exclude) || 
                                      doesSegmentIntersectComponent(mid2, p2, components, exclude);
                        
                        let chosenSegments = [];
                        if (coll1 && !coll2) {
                            // H-then-V collides, but V-then-H is clean. Choose V-then-H
                            chosenSegments = [
                                [p1, { x: p1.x, y: p2.y }],
                                [{ x: p1.x, y: p2.y }, p2]
                            ];
                        } else {
                            // Default to H-then-V
                            chosenSegments = [
                                [p1, { x: p2.x, y: p1.y }],
                                [{ x: p2.x, y: p1.y }, p2]
                            ];
                        }
                        
                        chosenSegments.forEach(seg => {
                            if (seg[0].x !== seg[1].x || seg[0].y !== seg[1].y) {
                                const routed = routeAroundComponent(seg[0], seg[1], components, exclude);
                                routed.forEach(rSeg => newWires.push(rSeg));
                            }
                        });
                    } else {
                        // Straight wire segment - check if it intersects and detour
                        const exclude = [];
                        components.forEach(c => {
                            const pins = getCompPins(c);
                            const hasP1 = pins.some(p => Math.abs(p.x - p1.x) < 1 && Math.abs(p.y - p1.y) < 1);
                            const hasP2 = pins.some(p => Math.abs(p.x - p2.x) < 1 && Math.abs(p.y - p2.y) < 1);
                            if (hasP1 || hasP2) exclude.push(c);
                        });
                        const routed = routeAroundComponent(p1, p2, components, exclude);
                        routed.forEach(rSeg => newWires.push(rSeg));
                    }
                } else {
                    newWires.push(wire);
                }
            });
            wires = newWires;

            // Merge contiguous collinear wire segments to keep connections clean (no messy recursive slices)
            let merged = true;
            while (merged) {
                merged = false;
                for (let i = 0; i < wires.length; i++) {
                    for (let j = i + 1; j < wires.length; j++) {
                        const w1 = wires[i];
                        const w2 = wires[j];
                        
                        let shared = null;
                        let other1 = null;
                        let other2 = null;
                        
                        const ptsMatch = (a, b) => Math.abs(a.x - b.x) < 1 && Math.abs(a.y - b.y) < 1;
                        
                        if (ptsMatch(w1[0], w2[0])) { shared = w1[0]; other1 = w1[1]; other2 = w2[1]; }
                        else if (ptsMatch(w1[0], w2[1])) { shared = w1[0]; other1 = w1[1]; other2 = w2[0]; }
                        else if (ptsMatch(w1[1], w2[0])) { shared = w1[1]; other1 = w1[0]; other2 = w2[1]; }
                        else if (ptsMatch(w1[1], w2[1])) { shared = w1[1]; other1 = w1[0]; other2 = w2[0]; }
                        
                        if (shared) {
                            const isH1 = Math.abs(w1[0].y - w1[1].y) < 1;
                            const isV1 = Math.abs(w1[0].x - w1[1].x) < 1;
                            const isH2 = Math.abs(w2[0].y - w2[1].y) < 1;
                            const isV2 = Math.abs(w2[0].x - w2[1].x) < 1;
                            
                            if ((isH1 && isH2 && Math.abs(other1.y - other2.y) < 1) || 
                                (isV1 && isV2 && Math.abs(other1.x - other2.x) < 1)) {
                                wires[i] = [other1, other2];
                                wires.splice(j, 1);
                                merged = true;
                                break;
                            }
                        }
                    }
                    if (merged) break;
                }
            }

            // Filter out any zero-length segments that might have resulted from movement/snapping
            wires = wires.filter(w => !(Math.abs(w[0].x - w[1].x) < 1 && Math.abs(w[0].y - w[1].y) < 1));

            // Verify no NEW overlaps were introduced by this drag.
            // Pre-existing overlaps (captured at drag-start) are exempt — the user
            // should always be able to drag overlapping components to resolve them.
            let overlapDetected = false;
            for (const sel of selectedComponents) {
                // Check for NEW component-component collisions
                for (const other of components) {
                    if (other === sel || selectedComponents.includes(other)) continue;
                    if (componentsOverlap(sel, other)) {
                        // Is this pair pre-existing?
                        const wasPreExisting = preExistingOverlapPairs.some(
                            ([a, b]) => (a === sel && b === other) || (a === other && b === sel)
                        );
                        if (!wasPreExisting) {
                            overlapDetected = true;
                            break;
                        }
                    }
                }
                if (overlapDetected) break;

                // Check for NEW component-wire collisions (only for moved/selected comps)
                for (const wire of wires) {
                    if (doesCompOverlapAnyWire(sel, [wire])) {
                        const wasPreExisting = preExistingWireOverlaps.some(
                            ([c, w]) => c === sel && w === wire
                        );
                        if (!wasPreExisting) {
                            overlapDetected = true;
                            break;
                        }
                    }
                }
                if (overlapDetected) break;
            }

            if (overlapDetected) {
                // Only rollback if we actually saved a state at the start of this drag.
                // Without this guard, a plain click on an already-overlapping component
                // would blindly pop a previous valid state (e.g. the just-imported circuit)
                // and make the canvas appear empty with no way to undo.
                if (hasSavedStateForThisDrag && undoStack.length > 0) {
                    const cleanState = undoStack.pop();
                    components = cleanState.components;
                    wires = cleanState.wires;
                    updateUndoRedoButtons();
                }
                document.getElementById("statusText").innerText = "⚠️ Drag cancelled: Overlaps another component or wire.";
                selectedComponents = [];
                selectedComp = null;
                selectedWires = [];
                selectedWirePts = [];
                isDragging = false;
                hasSavedStateForThisDrag = false;
                updatePropertiesPanel();
                render();
                return;
            }

            attachedWireEndpoints = [];

            updatePropertiesPanel();
            render();
        }
        isSelectingBox = false;
        isDragging = false;
        hasSavedStateForThisDrag = false;
        preExistingOverlapPairs = [];
        preExistingWireOverlaps = [];
    });

    // 5. Right Click (Cancel)
    canvas.addEventListener("contextmenu", (e) => {
        e.preventDefault();
        wireStart = null;
        selectedComp = null;
        selectedWires = [];
        mode = 'select';
        updateToolUI();
        updatePropertiesPanel();
        render();
    });

    // Keyboard shortcut listener
    document.addEventListener("keydown", (e) => {
        const active = document.activeElement;
        const isEditingInput = active && (active.tagName === 'INPUT' || active.tagName === 'SELECT' || active.tagName === 'TEXTAREA');

        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
            if (isEditingInput) return;
            e.preventDefault();
            undo();
            return;
        }
        if ((e.ctrlKey || e.metaKey) && (e.key.toLowerCase() === 'y' || (e.shiftKey && e.key.toLowerCase() === 'z'))) {
            if (isEditingInput) return;
            e.preventDefault();
            redo();
            return;
        }

        if (e.key === 'Delete') {
            if (isEditingInput) return;
            deleteSelectedItems();
        }
        if (e.key === 'Escape') {
            wireStart = null;
            selectedComponents = [];
            selectedComp = null;
            selectedWires = [];
            selectedWirePts = [];
            mode = 'select';
            
            // Clear interactive debug highlights
            highlightedComponents = [];
            highlightedNodes = [];
            highlightedNetlistLine = null;
            scheduleAnimation();
            updateNetlistPreview(lastNetlistText);

            updateToolUI();
            updatePropertiesPanel();
            render();
        }
        if (e.key.toLowerCase() === 'r') {
            if (isEditingInput) {
                return;
            }
            if (selectedComponents.length > 0) {
                let overlapDetected = false;
                selectedComponents.forEach(c => {
                    const tempComp = { ...c, rotation: ((c.rotation || 0) + 90) % 360 };
                    if (isOverlappingAny(tempComp, components, selectedComponents) || doesCompOverlapAnyWire(tempComp, wires)) {
                        overlapDetected = true;
                    }
                });
                if (overlapDetected) {
                    document.getElementById("statusText").innerText = "⚠️ Cannot rotate: Overlaps another component or wire.";
                    return;
                }

                saveState();
                selectedComponents.forEach(c => {
                    c.rotation = ((c.rotation || 0) + 90) % 360;
                });
                updatePropertiesPanel();
                render();
            } else if (mode !== 'select' && mode !== 'wire') {
                // Rotate ghost component before placement
                placementRotation = ((placementRotation || 0) + 90) % 360;
                render();
            }
        }
    });

    // ═══════════════════════════════════════════
    // HIT TESTING (Per-type bounding boxes)
    // ═══════════════════════════════════════════
    function hitTest(worldPos) {
        // Iterate in reverse so topmost (last drawn) is hit first
        for (let i = components.length - 1; i >= 0; i--) {
            const c = components[i];
            if (c.type === 'label') {
                const labelText = c.params && c.params.name ? c.params.name : c.name;
                const textWidth = Math.max(40, labelText.length * 8);
                const hw = textWidth / 2;
                const hh = 10;
                const centerY = c.y - 15;
                const hitText = (worldPos.x >= c.x - hw && worldPos.x <= c.x + hw &&
                                 worldPos.y >= centerY - hh && worldPos.y <= centerY + hh);
                const hitCenter = (worldPos.x >= c.x - 20 && worldPos.x <= c.x + 20 &&
                                   worldPos.y >= c.y - 10 && worldPos.y <= c.y + 10);
                if (hitText || hitCenter) {
                    return c;
                }
            } else {
                const db = COMPONENT_DB[c.type];
                const hb = db ? db.hitbox : { w: 40, h: 40 };
                const rot = c.rotation || 0;
                const hw = (rot === 90 || rot === 270) ? hb.h / 2 : hb.w / 2;
                const hh = (rot === 90 || rot === 270) ? hb.w / 2 : hb.h / 2;
                if (worldPos.x >= c.x - hw && worldPos.x <= c.x + hw &&
                    worldPos.y >= c.y - hh && worldPos.y <= c.y + hh) {
                    return c;
                }
            }
        }
        return null;
    }

    function getDistanceToSegment(p, p1, p2) {
        const A = p.x - p1.x;
        const B = p.y - p1.y;
        const C = p2.x - p1.x;
        const D = p2.y - p1.y;

        const dot = A * C + B * D;
        const lenSq = C * C + D * D;
        let param = -1;
        if (lenSq !== 0) {
            param = dot / lenSq;
        }

        let xx, yy;
        if (param < 0) {
            xx = p1.x;
            yy = p1.y;
        } else if (param > 1) {
            xx = p2.x;
            yy = p2.y;
        } else {
            xx = p1.x + param * C;
            yy = p1.y + param * D;
        }

        const dx = p.x - xx;
        const dy = p.y - yy;
        return Math.sqrt(dx * dx + dy * dy);
    }

    function hitTestWire(worldPos) {
        const threshold = 8;
        for (let i = 0; i < wires.length; i++) {
            const wire = wires[i];
            if (wire.length < 2) continue;
            const p1 = wire[0];
            const p2 = wire[1];
            
            if (p1.x !== p2.x && p1.y !== p2.y) {
                const mid = { x: p2.x, y: p1.y };
                const d1 = getDistanceToSegment(worldPos, p1, mid);
                const d2 = getDistanceToSegment(worldPos, mid, p2);
                if (d1 < threshold || d2 < threshold) {
                    return wire;
                }
            } else {
                const d = getDistanceToSegment(worldPos, p1, p2);
                if (d < threshold) {
                    return wire;
                }
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
        ctx.fillStyle = currentThemeColors.labelColor || "#E0E0E0";
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
        ctx.fillStyle = currentThemeColors.labelColor || "#E0E0E0";
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
        ctx.fillStyle = currentThemeColors.labelColor || "#E0E0E0";
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
        ctx.fillStyle = currentThemeColors.valueColor || "#FF9800";
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
        ctx.fillStyle = currentThemeColors.wireSelected || "#4FC1FF";
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

    function drawIC(ctx, sx, sy, z, comp) {
        const params    = (comp && comp.params) ? comp.params : {};
        const subcktName = params.subckt_name || 'IC';
        const numPins   = Math.max(2, parseInt(params.num_pins, 10) || 2);

        // DIP layout: split pins between left and right sides
        const leftCount  = Math.ceil(numPins / 2);
        const rightCount = Math.floor(numPins / 2);
        const vStep      = 20 * z;                             // one grid cell per pin
        const bodyH      = (Math.max(leftCount, rightCount) - 1) * vStep + 20 * z;
        const bodyW      = 60 * z;
        const leadLen    = 12 * z;  // stub lead from body edge to pin dot

        // IC body rectangle
        ctx.strokeRect(sx - bodyW / 2, sy - bodyH / 2, bodyW, bodyH);

        // Notch at top center
        ctx.beginPath();
        ctx.arc(sx, sy - bodyH / 2, 4 * z, 0, Math.PI);
        ctx.stroke();

        // Subcircuit name inside body
        ctx.fillStyle = currentThemeColors.labelColor || '#94a3b8';
        ctx.font = `bold ${Math.min(11, 9 + z) * z}px 'Inter', 'Segoe UI', Arial`;
        ctx.textAlign    = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(subcktName, sx, sy);

        // Left-side pins (top to bottom)
        const topY = sy - (leftCount - 1) * vStep / 2;
        for (let i = 0; i < leftCount; i++) {
            const py = topY + i * vStep;
            const px = sx - bodyW / 2;

            ctx.beginPath();
            ctx.moveTo(px - leadLen, py);
            ctx.lineTo(px, py);
            ctx.stroke();
            drawPinDot(ctx, px - leadLen, py, z);

            // Pin number label
            ctx.fillStyle = currentThemeColors.valueColor || '#f97316';
            ctx.font = `${9 * z}px 'Inter', Arial`;
            ctx.textAlign    = 'left';
            ctx.textBaseline = 'middle';
            ctx.fillText(String(i + 1), px + 3 * z, py);
        }

        // Right-side pins (bottom to top, DIP convention)
        const rightTopY = sy - (rightCount - 1) * vStep / 2;
        for (let i = 0; i < rightCount; i++) {
            const pinNum = numPins - i;  // pin numbers continue from bottom-right
            const py = rightTopY + (rightCount - 1 - i) * vStep;
            const px = sx + bodyW / 2;

            ctx.beginPath();
            ctx.moveTo(px, py);
            ctx.lineTo(px + leadLen, py);
            ctx.stroke();
            drawPinDot(ctx, px + leadLen, py, z);

            // Pin number label
            ctx.fillStyle = currentThemeColors.valueColor || '#f97316';
            ctx.font = `${9 * z}px 'Inter', Arial`;
            ctx.textAlign    = 'right';
            ctx.textBaseline = 'middle';
            ctx.fillText(String(pinNum), px - 3 * z, py);
        }
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
        ctx.fillStyle = currentThemeColors.wireSelected || "#4FC1FF";
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
        ctx.fillStyle = currentThemeColors.wireColor || "#4FC1FF";
        ctx.beginPath();
        ctx.arc(x, y, 2.5 * z, 0, Math.PI * 2);
        ctx.fill();
    }

    // Fallback renderer for unknown component types
    function drawFallback(ctx, sx, sy, z, type) {
        const size = 30 * z;
        ctx.strokeRect(sx - size / 2, sy - size / 2, size, size);
        ctx.fillStyle = currentThemeColors.labelColor || "#E0E0E0";
        ctx.font = `${10 * z}px Arial`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(type, sx, sy);
    }

    // ═══════════════════════════════════════════
    // MAIN RENDERING ENGINE
    // ═══════════════════════════════════════════
    function render() {
        const style = getComputedStyle(document.body);
        currentThemeColors.canvasBg = style.getPropertyValue('--canvas-bg').trim() || "#090d16";
        currentThemeColors.gridColor = style.getPropertyValue('--grid-color').trim() || "rgba(255, 255, 255, 0.08)";
        currentThemeColors.wireColor = style.getPropertyValue('--wire-color').trim() || "#4fc1ff";
        currentThemeColors.wireSelected = style.getPropertyValue('--wire-selected').trim() || "#38bdf8";
        currentThemeColors.componentColor = style.getPropertyValue('--component-color').trim() || "#cbd5e1";
        currentThemeColors.labelColor = style.getPropertyValue('--label-color').trim() || "#94a3b8";
        currentThemeColors.valueColor = style.getPropertyValue('--value-color').trim() || "#f97316";

        // Clear Canvas
        ctx.fillStyle = currentThemeColors.canvasBg;
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw Grid dots
        ctx.fillStyle = currentThemeColors.gridColor;
        const step = gridSize * zoom;
        if (step > 5) {
            const startX = offsetX % step;
            const startY = offsetY % step;
            for (let x = startX; x < canvas.width; x += step) {
                for (let y = startY; y < canvas.height; y += step) {
                    ctx.fillRect(x, y, 1.5, 1.5);
                }
            }
        }

        // Draw Wires
        wires.forEach(wire => {
            if (wire.length < 2) return;
            const p1 = worldToScreen(wire[0].x, wire[0].y);
            const p2 = worldToScreen(wire[1].x, wire[1].y);

            ctx.save();
            
            // Check if wire belongs to a highlighted node
            const isNodeHighlighted = highlightedNodes.length > 0 && 
                (getNodeAt(wire[0]) === highlightedNodes[0] || getNodeAt(wire[1]) === highlightedNodes[0]);

            if (isNodeHighlighted) {
                const pulse = 1 + 2 * Math.sin(Date.now() / 150);
                ctx.strokeStyle = "rgba(0, 255, 127, 0.85)"; // Spring Green
                ctx.shadowColor = "rgba(0, 255, 127, 0.6)";
                ctx.shadowBlur = 8 + pulse;
                ctx.lineWidth = Math.max(3.5, 5 * zoom);
            } else if (selectedWires.includes(wire)) {
                ctx.strokeStyle = currentThemeColors.wireSelected;
                ctx.lineWidth = Math.max(2, 3.5 * zoom);
            } else {
                ctx.strokeStyle = currentThemeColors.wireColor;
                ctx.lineWidth = Math.max(1, 2 * zoom);
            }

            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            // Force orthogonal routing visually if diagonal
            if (wire[0].x !== wire[1].x && wire[0].y !== wire[1].y) {
                ctx.lineTo(p2.x, p1.y);
            }
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();

            // Wire endpoint dots
            if (selectedWires.includes(wire)) {
                ctx.fillStyle = currentThemeColors.wireSelected;
            } else {
                ctx.fillStyle = currentThemeColors.wireColor;
            }
            ctx.beginPath(); ctx.arc(p1.x, p1.y, (selectedWires.includes(wire) ? 4 : 3) * zoom, 0, Math.PI * 2); ctx.fill();
            ctx.beginPath(); ctx.arc(p2.x, p2.y, (selectedWires.includes(wire) ? 4 : 3) * zoom, 0, Math.PI * 2); ctx.fill();
            ctx.restore();
        });

        // Draw Components
        components.forEach(comp => {
            const pos = worldToScreen(comp.x, comp.y);
            if (comp.type === 'junction') {
                ctx.save();
                ctx.fillStyle = currentThemeColors.wireColor || "#4fc1ff";
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, 4 * zoom, 0, Math.PI * 2);
                ctx.fill();
                ctx.restore();
                return;
            }

            const SKIP_RENDER_TYPES = ['wire', 'junction', 'crossover', 'terminal', 'text'];
            if (SKIP_RENDER_TYPES.includes(comp.type)) return;

            if (comp.type === 'label') {
                const labelText = comp.params && comp.params.name ? comp.params.name : comp.name;
                ctx.fillStyle = currentThemeColors.valueColor || "#FFC107";
                ctx.font = `bold ${14 * zoom}px 'Segoe UI', Arial`;
                ctx.textAlign = "center";
                ctx.textBaseline = "bottom";
                ctx.fillText(labelText, pos.x, pos.y - 8 * zoom);

                // Draw selection highlight for label text
                if (selectedComponents.includes(comp)) {
                    ctx.save();
                    ctx.strokeStyle = currentThemeColors.wireSelected || "rgba(0, 229, 255, 0.85)";
                    ctx.lineWidth = 1.5;
                    ctx.setLineDash([3, 2]);
                    const textWidth = ctx.measureText(labelText).width;
                    const padding = 4 * zoom;
                    ctx.strokeRect(
                        pos.x - textWidth / 2 - padding,
                        pos.y - 24 * zoom - padding,
                        textWidth + padding * 2,
                        18 * zoom + padding * 2
                    );
                    ctx.restore();
                }

                // Pulse glow for locate/highlight action
                if (highlightedComponents.includes(comp)) {
                    ctx.save();
                    const textWidth = ctx.measureText(labelText).width;
                    const padding = 4 * zoom;
                    const pulse = 2 + 2 * Math.sin(Date.now() / 150);
                    ctx.strokeStyle = "rgba(255, 64, 129, 0.85)";
                    ctx.shadowColor = "rgba(255, 64, 129, 0.6)";
                    ctx.shadowBlur = 8 + pulse;
                    ctx.lineWidth = 2.5;
                    ctx.strokeRect(
                        pos.x - textWidth / 2 - padding - pulse,
                        pos.y - 24 * zoom - padding - pulse,
                        textWidth + (padding + pulse) * 2,
                        18 * zoom + (padding + pulse) * 2
                    );
                    ctx.restore();
                }
                return;
            }
            // Selection highlight
            if (selectedComponents.includes(comp)) {
                const db = COMPONENT_DB[comp.type];
                const hb = db ? db.hitbox : { w: 40, h: 40 };
                const rot = comp.rotation || 0;
                const w = (rot === 90 || rot === 270) ? hb.h : hb.w;
                const h = (rot === 90 || rot === 270) ? hb.w : hb.h;
                ctx.save();
                ctx.strokeStyle = currentThemeColors.wireSelected || "rgba(0, 229, 255, 0.85)";
                ctx.lineWidth = 2;
                ctx.setLineDash([4, 3]);
                ctx.strokeRect(
                    pos.x - (w / 2 + 6) * zoom,
                    pos.y - (h / 2 + 6) * zoom,
                    (w + 12) * zoom,
                    (h + 12) * zoom
                );
                ctx.restore();
            }

            // Pulse glow for locate/highlight action
            if (highlightedComponents.includes(comp)) {
                const db = COMPONENT_DB[comp.type];
                const hb = db ? db.hitbox : { w: 40, h: 40 };
                const rot = comp.rotation || 0;
                const w = (rot === 90 || rot === 270) ? hb.h : hb.w;
                const h = (rot === 90 || rot === 270) ? hb.w : hb.h;
                const pulse = 4 + 4 * Math.sin(Date.now() / 150);
                ctx.save();
                ctx.strokeStyle = "rgba(255, 64, 129, 0.85)";
                ctx.shadowColor = "rgba(255, 64, 129, 0.6)";
                ctx.shadowBlur = 10 + pulse;
                ctx.lineWidth = 3;
                ctx.strokeRect(
                    pos.x - (w / 2 + pulse) * zoom,
                    pos.y - (h / 2 + pulse) * zoom,
                    (w + pulse * 2) * zoom,
                    (h + pulse * 2) * zoom
                );
                ctx.restore();
            }

            // Draw the schematic symbol
            ctx.strokeStyle = currentThemeColors.componentColor || "#E0E0E0";
            ctx.lineWidth = Math.max(1, 2 * zoom);
            const renderer = SYMBOL_RENDERERS[comp.type];
            ctx.save();
            ctx.translate(pos.x, pos.y);
            ctx.rotate((comp.rotation || 0) * Math.PI / 180);
            if (renderer) {
                renderer(ctx, 0, 0, zoom, comp);
            } else {
                drawFallback(ctx, 0, 0, zoom, comp.type);
            }
            ctx.restore();

            // Draw component name label (above)
            ctx.fillStyle = currentThemeColors.labelColor || "#E0E0E0";
            ctx.font = `bold ${12 * zoom}px 'Segoe UI', Arial`;
            ctx.textAlign = "center";
            ctx.textBaseline = "bottom";
            const db = COMPONENT_DB[comp.type];
            const rot = comp.rotation || 0;
            const h_rotated = (rot === 90 || rot === 270) ? (db ? db.hitbox.w : 40) : (db ? db.hitbox.h : 40);
            const labelOffY = h_rotated / 2 + 14;
            ctx.fillText(comp.name, pos.x, pos.y - labelOffY * zoom);

            // Draw value label (below)
            const mainValue = comp.value || getDisplayValue(comp);
            if (mainValue && comp.type !== 'ground') {
                ctx.fillStyle = currentThemeColors.valueColor || "#FF9800";
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
                        ctx.fillStyle = currentThemeColors.wireSelected;
                        ctx.strokeStyle = currentThemeColors.wireSelected;
                        ctx.lineWidth = 2;
                        ctx.beginPath();
                        ctx.arc(sp.x, sp.y, 7 * zoom, 0, Math.PI * 2);
                        ctx.stroke();
                        ctx.beginPath();
                        ctx.arc(sp.x, sp.y, 3.5 * zoom, 0, Math.PI * 2);
                        ctx.fill();
                    } else {
                        // Subtle pin markers for unconnected pins
                        ctx.fillStyle = currentThemeColors.wireColor + "4D"; // 30% opacity
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
            ctx.save();
            ctx.translate(gPos.x, gPos.y);
            ctx.rotate((placementRotation || 0) * Math.PI / 180);
            if (renderer) {
                // Pass a ghost comp so param-aware renderers (e.g. drawIC) can render defaults
                const ghostComp = { type: mode, params: (COMPONENT_DB[mode] ? { ...COMPONENT_DB[mode].params } : {}) };
                renderer(ctx, 0, 0, zoom, ghostComp);
            }
            ctx.restore();
            ctx.globalAlpha = 1.0;
        }

        // Draw selected wire endpoints (junctions/corners)
        selectedWirePts.forEach(pt => {
            const sp = worldToScreen(pt.x, pt.y);
            ctx.save();
            ctx.fillStyle = "rgba(0, 229, 255, 0.85)";
            ctx.strokeStyle = "#F8FAFC";
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(sp.x, sp.y, 6 * zoom, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
            ctx.restore();
        });

        // Draw active selection box (marquee selection)
        if (isSelectingBox && selectionStart && selectionEnd) {
            const p1 = worldToScreen(selectionStart.x, selectionStart.y);
            const p2 = worldToScreen(selectionEnd.x, selectionEnd.y);

            ctx.save();
            ctx.strokeStyle = "rgba(0, 229, 255, 0.6)";
            ctx.fillStyle = "rgba(0, 229, 255, 0.08)";
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            ctx.fillRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
            ctx.strokeRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
            ctx.restore();
        }

        // Draw pulsing dots for pins on highlighted node
        if (highlightedNodes.length > 0) {
            components.forEach(comp => {
                const pins = getCompPins(comp);
                pins.forEach(pin => {
                    if (getNodeAt(pin) === highlightedNodes[0]) {
                        const pos = worldToScreen(pin.x, pin.y);
                        const pulse = 2 + 2 * Math.sin(Date.now() / 100);
                        ctx.save();
                        ctx.fillStyle = "rgba(0, 255, 127, 0.9)";
                        ctx.shadowColor = "rgba(0, 255, 127, 0.8)";
                        ctx.shadowBlur = 10 + pulse;
                        ctx.beginPath();
                        ctx.arc(pos.x, pos.y, (5 + pulse) * zoom, 0, Math.PI * 2);
                        ctx.fill();
                        ctx.restore();
                    }
                });
            });
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
    function deleteSelectedItems() {
        if (selectedComponents.length > 0 || selectedWires.length > 0) {
            saveState();
        }
        if (selectedComponents.length > 0) {
            selectedComponents.forEach(comp => {
                const idx = components.indexOf(comp);
                if (idx !== -1) components.splice(idx, 1);
            });
            selectedComponents = [];
            selectedComp = null;
        }
        if (selectedWires.length > 0) {
            selectedWires.forEach(wire => {
                const idx = wires.indexOf(wire);
                if (idx !== -1) wires.splice(idx, 1);
            });
            selectedWires = [];
        }
        selectedWirePts = [];
        updatePropertiesPanel();
        render();
    }

    function updatePropertiesPanel() {
        const panel = document.getElementById("propertiesPanel");
        panel.innerHTML = '';

        const hasComps = selectedComponents.length > 0;
        const hasWires = selectedWires.length > 0;

        if (!hasComps && !hasWires) {
            panel.innerHTML = '<p class="placeholder-text">Select a component or wire</p>';
            return;
        }

        if (hasComps && hasWires) {
            const badge = document.createElement('div');
            badge.className = 'prop-type-badge';
            badge.textContent = 'Multiple Items';
            panel.appendChild(badge);

            const idTitle = document.createElement('p');
            idTitle.className = 'prop-section-title';
            idTitle.textContent = 'Selection Summary';
            panel.appendChild(idTitle);

            addPropField(panel, 'Components', String(selectedComponents.length), null, true);
            addPropField(panel, 'Wires', String(selectedWires.length), null, true);

            const delBtn = document.createElement('button');
            delBtn.className = 'prop-delete-btn';
            delBtn.textContent = '🗑 Delete Selected';
            delBtn.addEventListener('click', deleteSelectedItems);
            panel.appendChild(delBtn);
            return;
        }

        if (hasWires) {
            if (selectedWires.length === 1) {
                const wire = selectedWires[0];
                const badge = document.createElement('div');
                badge.className = 'prop-type-badge';
                badge.textContent = 'Wire Connection';
                panel.appendChild(badge);

                const idTitle = document.createElement('p');
                idTitle.className = 'prop-section-title';
                idTitle.textContent = 'Coordinates';
                panel.appendChild(idTitle);

                addPropField(panel, 'Start X', String(wire[0].x), null, true);
                addPropField(panel, 'Start Y', String(wire[0].y), null, true);
                addPropField(panel, 'End X', String(wire[1].x), null, true);
                addPropField(panel, 'End Y', String(wire[1].y), null, true);
            } else {
                const badge = document.createElement('div');
                badge.className = 'prop-type-badge';
                badge.textContent = 'Multiple Wires';
                panel.appendChild(badge);

                const idTitle = document.createElement('p');
                idTitle.className = 'prop-section-title';
                idTitle.textContent = 'Selection';
                panel.appendChild(idTitle);

                addPropField(panel, 'Wires Count', String(selectedWires.length), null, true);
            }

            const delBtn = document.createElement('button');
            delBtn.className = 'prop-delete-btn';
            delBtn.textContent = '🗑 Delete Selected';
            delBtn.addEventListener('click', deleteSelectedItems);
            panel.appendChild(delBtn);
            return;
        }

        if (selectedComponents.length > 1) {
            const badge = document.createElement('div');
            badge.className = 'prop-type-badge';
            badge.textContent = 'Multiple Components';
            panel.appendChild(badge);

            const idTitle = document.createElement('p');
            idTitle.className = 'prop-section-title';
            idTitle.textContent = 'Selection';
            panel.appendChild(idTitle);

            addPropField(panel, 'Components Count', String(selectedComponents.length), null, true);

            const delBtn = document.createElement('button');
            delBtn.className = 'prop-delete-btn';
            delBtn.textContent = '🗑 Delete Selected';
            delBtn.addEventListener('click', deleteSelectedItems);
            panel.appendChild(delBtn);
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
            saveState();
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

            // ─── Special IC Configuration panel ──────────────────────────────
            if (comp.type === 'ic') {
                const icTitle = document.createElement('p');
                icTitle.className = 'prop-section-title';
                icTitle.textContent = '🔌 IC Configuration';
                panel.appendChild(icTitle);

                // Subcircuit name
                addPropField(panel, 'Subckt Name', String(comp.params.subckt_name || 'MyIC'), (newVal) => {
                    saveState();
                    comp.params.subckt_name = newVal.trim() || 'MyIC';
                    render();
                });

                // Pin count — triggers immediate re-render to show new pin layout
                addPropField(panel, 'Pin Count', String(comp.params.num_pins || '2'), (newVal) => {
                    const n = Math.max(2, parseInt(newVal, 10) || 2);
                    saveState();
                    comp.params.num_pins = String(n);
                    render();
                    updatePropertiesPanel();  // refresh so label shows new value
                });

                // Custom .subckt body (multiline textarea)
                const subcktRow = document.createElement('div');
                subcktRow.className = 'prop-row';
                subcktRow.style.flexDirection = 'column';
                subcktRow.style.alignItems = 'stretch';
                subcktRow.style.gap = '4px';

                const subcktLbl = document.createElement('span');
                subcktLbl.className = 'prop-label';
                subcktLbl.textContent = 'Custom .subckt';
                subcktRow.appendChild(subcktLbl);

                const subcktHint = document.createElement('span');
                subcktHint.style.cssText = 'font-size:10px;color:var(--text-dim,#64748b);line-height:1.3;';
                subcktHint.textContent = 'Paste a full ngspice .subckt block here. Leave empty to rely on an external file.';
                subcktRow.appendChild(subcktHint);

                const subcktTA = document.createElement('textarea');
                subcktTA.className = 'prop-input';
                subcktTA.rows = 5;
                subcktTA.style.cssText = 'font-family:monospace;font-size:11px;resize:vertical;width:100%;box-sizing:border-box;';
                subcktTA.value = comp.params.custom_subckt || '';
                subcktTA.addEventListener('change', (e) => {
                    saveState();
                    comp.params.custom_subckt = e.target.value;
                });
                subcktRow.appendChild(subcktTA);
                panel.appendChild(subcktRow);

            } else {
                // ─── Generic SPICE Parameters block (all other types) ─────────
                const paramTitle = document.createElement('p');
                paramTitle.className = 'prop-section-title';
                paramTitle.textContent = 'SPICE Parameters';
                panel.appendChild(paramTitle);

                for (const [key, val] of Object.entries(comp.params)) {
                    addPropField(panel, key, String(val), (newVal) => {
                        saveState();
                        comp.params[key] = newVal;
                        // Sync the top-level value field
                        if (key === 'value' || key === 'dc' || key === 'mag') {
                            comp.value = newVal;
                        }
                        render();
                    });
                }
            }
        }


        // Section: Orientation
        const orientTitle = document.createElement('p');
        orientTitle.className = 'prop-section-title';
        orientTitle.textContent = 'Orientation';
        panel.appendChild(orientTitle);

        const rotRow = document.createElement('div');
        rotRow.className = 'prop-row';

        const rotLbl = document.createElement('span');
        rotLbl.className = 'prop-label';
        rotLbl.textContent = 'Rotation';
        rotRow.appendChild(rotLbl);

        const rotSel = document.createElement('select');
        rotSel.className = 'prop-input';
        [0, 90, 180, 270].forEach(deg => {
            const opt = document.createElement('option');
            opt.value = deg;
            opt.textContent = `${deg}°`;
            if ((comp.rotation || 0) === deg) {
                opt.selected = true;
            }
            rotSel.appendChild(opt);
        });
        rotSel.addEventListener('change', (e) => {
            const newRot = parseInt(e.target.value);
            const tempComp = { ...comp, rotation: newRot };
            if (isOverlappingAny(tempComp, components, [comp]) || doesCompOverlapAnyWire(tempComp, wires)) {
                document.getElementById("statusText").innerText = "⚠️ Cannot rotate: Overlaps another component or wire.";
                rotSel.value = comp.rotation || 0;
                return;
            }
            saveState();
            comp.rotation = newRot;
            render();
        });
        rotRow.appendChild(rotSel);
        panel.appendChild(rotRow);

        // Delete button
        const delBtn = document.createElement('button');
        delBtn.className = 'prop-delete-btn';
        delBtn.textContent = '🗑 Delete Selected';
        delBtn.addEventListener('click', deleteSelectedItems);
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
            input.addEventListener('change', (e) => onChange(e.target.value));
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
            placementRotation = 0;

            // Reset select dropdowns if a standard button was clicked
            document.querySelectorAll('.toolbar-select').forEach(sel => {
                sel.value = '';
            });

            updateToolUI();
            render();
        });
    });

    const btnReroute = document.getElementById('btn-reroute');
    if (btnReroute) {
        btnReroute.addEventListener('click', () => {
            saveState();
            rerouteAllWires();
        });
    }

    const btnUndo = document.getElementById('btn-undo');
    if (btnUndo) {
        btnUndo.addEventListener('click', () => {
            undo();
        });
    }

    const btnRedo = document.getElementById('btn-redo');
    if (btnRedo) {
        btnRedo.addEventListener('click', () => {
            redo();
        });
    }

    document.querySelectorAll('.toolbar-select').forEach(sel => {
        sel.addEventListener('change', (e) => {
            mode = e.target.value;
            wireStart = null;
            placementRotation = 0;

            // Reset other select dropdowns
            document.querySelectorAll('.toolbar-select').forEach(other => {
                if (other !== e.target) other.value = '';
            });

            updateToolUI();
            render();
        });
    });

    function updateToolUI() {
        document.querySelectorAll('.btn-tool').forEach(btn => btn.classList.remove('active'));
        const activeBtn = document.getElementById(`tool-${mode}`);
        if (activeBtn) activeBtn.classList.add('active');

        // Highlight dropdown categories if their value is active
        document.querySelectorAll('.toolbar-select').forEach(sel => {
            let found = false;
            Array.from(sel.options).forEach(opt => {
                if (opt.value === mode && mode !== '') {
                    sel.value = mode;
                    found = true;
                }
            });
            if (found) {
                sel.classList.add('active');
            } else {
                if (sel.value === mode) {
                    sel.value = '';
                }
                sel.classList.remove('active');
            }
        });
    }

    // ═══════════════════════════════════════════
    // NETLIST PANEL TOGGLE & EDIT MODE
    // ═══════════════════════════════════════════
    const netlistToggle = document.getElementById("netlistToggle");
    const netlistBtnEdit = document.getElementById("netlistBtnEdit");
    const netlistViewerWrap = document.getElementById("netlistViewerWrap");
    const netlistEditable = document.getElementById("netlistEditable");
    const netlistManualWarning = document.getElementById("netlistManualWarning");
    const netlistBtnReset = document.getElementById("netlistBtnReset");

    if (netlistToggle) {
        netlistToggle.addEventListener('click', (e) => {
            // Do not toggle collapse when clicking Edit button or action bar
            if (e.target.closest('.netlist-header-actions') || e.target.closest('#netlistBtnEdit')) {
                return;
            }
            const panel = document.getElementById("netlistPreview");
            panel.classList.toggle('collapsed');
            const icon = netlistToggle.querySelector('.toggle-icon');
            icon.textContent = panel.classList.contains('collapsed') ? '▶' : '▼';
        });
    }

    if (netlistBtnEdit) {
        netlistBtnEdit.addEventListener('click', (e) => {
            e.stopPropagation();
            toggleNetlistEditMode();
        });
    }

    if (netlistEditable) {
        netlistEditable.addEventListener('input', () => {
            lastNetlistText = netlistEditable.value;
        });
    }

    if (netlistBtnReset) {
        netlistBtnReset.addEventListener('click', (e) => {
            e.stopPropagation();
            syncNetlistToCanvas();
        });
    }

    function toggleNetlistEditMode(forceState) {
        isManualNetlist = (forceState !== undefined) ? forceState : !isManualNetlist;

        if (isManualNetlist) {
            netlistViewerWrap.style.display = 'none';
            netlistEditable.style.display = 'block';
            netlistEditable.value = lastNetlistText;
            netlistBtnEdit.textContent = '🔒 View';
            netlistBtnEdit.classList.add('active');
            netlistManualWarning.style.display = 'flex';
        } else {
            netlistViewerWrap.style.display = 'block';
            netlistEditable.style.display = 'none';
            netlistBtnEdit.textContent = '✏️ Edit';
            netlistBtnEdit.classList.remove('active');
            netlistManualWarning.style.display = 'none';
            updateNetlistPreview(lastNetlistText);
        }
    }

    function syncNetlistToCanvas() {
        // Turn off manual netlist mode
        toggleNetlistEditMode(false);
        // Trigger simulation to regenerate netlist
        runSimulation(simConfig);
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
                    params: c.params || {}, rotation: c.rotation || 0
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
                nodeMap = data.node_map || {};
                
                // Refresh dropdowns
                populateSelect('sigNodeSelect', availableNodes);
                populateSelect('sigSourceSelect', availableSources);
                // Rebuild form to refresh sweepable selects
                rebuildSimForm();
            } else {
                nodeMap = {};
                if (data.logs) {
                    pushLogs(data.logs);
                }
            }
        } catch (e) {
            console.warn('Could not solve nodes:', e);
            nodeMap = {};
            pushLogs([{
                type: 'error',
                message: `Could not connect to backend to solve nodes: ${e.message || e}`,
                source: 'backend'
            }]);
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

    function checkGroundConnectedLabels() {
        const grid = 20;
        const adj = {};
        const addEdge = (u, v) => {
            if (!adj[u]) adj[u] = new Set();
            if (!adj[v]) adj[v] = new Set();
            adj[u].add(v);
            adj[v].add(u);
        };

        const snap = (x, y) => {
            return `${Math.round(x / grid) * grid},${Math.round(y / grid) * grid}`;
        };

        // 1. Build wire adjacency
        wires.forEach(wire => {
            if (wire.length < 2) return;
            const p1 = wire[0];
            const p2 = wire[1];

            const ax = Math.round(p1.x / grid) * grid;
            const ay = Math.round(p1.y / grid) * grid;
            const bx = Math.round(p2.x / grid) * grid;
            const by = Math.round(p2.y / grid) * grid;

            const dx = bx - ax;
            const dy = by - ay;

            const steps_x = dx !== 0 ? Math.floor(Math.abs(dx) / grid) : 0;
            const steps_y = dy !== 0 ? Math.floor(Math.abs(dy) / grid) : 0;
            const steps = Math.max(steps_x, steps_y);

            const p1Key = `${ax},${ay}`;
            if (!adj[p1Key]) adj[p1Key] = new Set();

            if (steps === 0) return;

            let prev = p1Key;
            for (let i = 1; i <= steps; i++) {
                const t = i / steps;
                const ix = Math.round((ax + dx * t) / grid) * grid;
                const iy = Math.round((ay + dy * t) / grid) * grid;
                const curr = `${ix},${iy}`;

                addEdge(prev, curr);
                prev = curr;
            }
        });

        // 2. Identify ground coordinates and label pins
        const groundCoords = new Set();
        const labelPins = [];

        components.forEach(comp => {
            const pins = getCompPins(comp);
            if (comp.type === 'ground') {
                pins.forEach(pin => {
                    groundCoords.add(snap(pin.x, pin.y));
                });
            } else if (comp.type === 'label') {
                const labelName = comp.params?.name || comp.name || 'LBL';
                pins.forEach(pin => {
                    labelPins.push({ name: labelName, pin: snap(pin.x, pin.y) });
                });
            }
        });

        if (groundCoords.size === 0 || labelPins.length === 0) {
            return [];
        }

        // 3. DFS reachable coordinates from ground
        const visited = new Set();
        const groundConnectedNodes = new Set();

        const dfs = (start) => {
            const stack = [start];
            while (stack.length > 0) {
                const curr = stack.pop();
                if (visited.has(curr)) continue;
                visited.add(curr);
                groundConnectedNodes.add(curr);

                const neighbors = adj[curr];
                if (neighbors) {
                    for (const neighbor of neighbors) {
                        if (!visited.has(neighbor)) {
                            stack.push(neighbor);
                        }
                    }
                }
            }
        };

        groundCoords.forEach(gCoord => {
            dfs(gCoord);
        });

        // 4. Collect warnings
        const warnings = [];
        labelPins.forEach(item => {
            if (groundConnectedNodes.has(item.pin)) {
                warnings.push(`Warning: Node label '${item.name}' is directly connected to a ground node.`);
            }
        });

        return warnings;
    }

    // ═══════════════════════════════════════════
    // SIMULATION EXECUTION
    // ═══════════════════════════════════════════
    document.getElementById('btnModalRun').addEventListener('click', async () => {
        // Run ground connection warning check on the frontend
        const warnings = checkGroundConnectedLabels();
        if (warnings.length > 0) {
            const confirmMsg = warnings.join('\n') + '\n\nDo you want to proceed with the simulation?';
            if (!confirm(confirmMsg)) {
                return; // User canceled simulation run
            }
        }

        const config = collectSimConfig();
        closeSimModal();
        await runSimulation(config);
    });

    function attachRawOutputDetailsListener(container) {
        const detailsElements = container.querySelectorAll('.raw-output-details');
        detailsElements.forEach(details => {
            details.addEventListener('toggle', async () => {
                if (details.open) {
                    const pre = details.querySelector('.raw-output-pre');
                    if (pre && pre.getAttribute('data-loaded') !== 'true') {
                        try {
                            pre.textContent = 'Loading log...';
                            const resp = await fetch('http://127.0.0.1:8000/api/simulation_log');
                            const logData = await resp.json();
                            if (logData.content) {
                                pre.textContent = logData.content;
                                pre.setAttribute('data-loaded', 'true');
                            } else {
                                pre.textContent = logData.error || 'Could not load log.';
                            }
                        } catch (err) {
                            pre.textContent = `Error fetching log: ${err.message || err}`;
                        }
                    }
                }
            });
        });
    }

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
                params: c.params || {}, rotation: c.rotation || 0
            })),
            wires: wires.map(w => [
                { x: w[0].x, y: w[0].y },
                { x: w[1].x, y: w[1].y }
            ]),
            simConfig: config,
            custom_netlist: isManualNetlist ? lastNetlistText : null
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
                updateNetlistPreview(data.netlist);
                nodeMap = data.node_map || {};
                pushLogs(data.logs || []);

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

                // Show raw output (collapsed by default, only shown/fetched on request)
                outputHtml += `<details class="raw-output-details" style="margin-top:8px;"><summary style="cursor:pointer;color:var(--text-secondary);font-size:12px;user-select:none;">Raw ngspice output</summary><pre class="raw-output-pre">Click to load raw output...</pre></details>`;

                if (!outputHtml) {
                    outputHtml = '<p class="sim-success">Simulation completed successfully.</p>';
                }

                simOutput.innerHTML = outputHtml;
                attachRawOutputDetailsListener(simOutput);
            } else {
                statusText.innerText = '❌ Simulation failed.';
                updateNetlistPreview(data.netlist);
                nodeMap = data.node_map || {};
                pushLogs(data.logs || []);
                
                simOutput.innerHTML = `<p class="sim-error">${escapeHtml(data.message || 'Unknown error')}</p>`;
                simOutput.innerHTML += `<details class="raw-output-details" style="margin-top:8px;"><summary style="cursor:pointer;color:var(--text-secondary);font-size:12px;user-select:none;">Raw ngspice output</summary><pre class="raw-output-pre">Click to load raw output...</pre></details>`;
                attachRawOutputDetailsListener(simOutput);
            }
        } catch (err) {
            console.error('Simulation request failed:', err);
            statusText.innerText = '❌ Connection error.';
            simOutput.innerHTML = `<p class="sim-error">Could not reach backend at http://127.0.0.1:8000. Is the server running?</p>`;
            nodeMap = {};
            pushLogs([{
                type: 'error',
                message: `Connection error: Could not reach backend at http://127.0.0.1:8000.`,
                source: 'backend'
            }]);
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
    window.openLightbox = function (src) {
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
                const rot = c.rotation || 0;
                const w = (rot === 90 || rot === 270) ? hb.h : hb.w;
                const h = (rot === 90 || rot === 270) ? hb.w : hb.h;
                // Inflate slightly to prevent wires grazing the bounding box
                const hw = (w / 2) + gSize * 0.5;
                const hh = (h / 2) + gSize * 0.5;
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
            const prev = path[i - 1];
            const curr = path[i];
            const next = path[i + 1];

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
    // AI DETECTION PREVIEW — INTERACTIVE EDITOR
    // ═══════════════════════════════════════════

    // --- Preview editor state ---
    const aiPreview = {
        components: [],      // raw from API (all types)
        connections: [],      // editable copy: [{pin1:{comp_idx,pin_id}, pin2:{comp_idx,pin_id}}, ...]
        pinAnchors: [],       // [{comp_idx, pin_id, x, y}, ...] — pixel coords from backend
        imageElement: null,   // loaded Image for canvas background
        imageW: 0,
        imageH: 0,
    };

    const aiEditor = {
        tool: 'select',        // 'select' | 'connect' | 'delete' | 'junction'
        pendingPin: null,      // {comp_idx, pin_id} — first pin of a new connection
        hoveredPin: null,      // {comp_idx, pin_id, x, y} or null
        hoveredConnIdx: null,  // index into aiPreview.connections or null
        hoveredJunction: null, // hovered virtual junction pin in delete mode
        pan: { x: 0, y: 0 },
        zoom: 1,
        isMouseDown: false,
        isDragging: false,
        isDraggingJunction: false,
        draggedJunction: null,
        dragDistance: 0,
        dragStart: { x: 0, y: 0 },
        panStart: { x: 0, y: 0 },
        mouseX: 0,            // raw canvas-local mouse coords
        mouseY: 0,
    };

    // DOM refs for the editor
    const aiCanvas = document.getElementById('aiDebugCanvas');
    const aiCtx = aiCanvas.getContext('2d');
    const aiCanvasWrap = document.getElementById('aiDebugCanvasWrap');
    const aiInfoEl = document.getElementById('aiDebugInfo');
    const aiStatsEl = document.getElementById('aiDebugStats');

    // --- Color generation (HSV-based, matching backend) ---
    function hsvToRgb(h, s, v) {
        // h: 0-360, s: 0-1, v: 0-1
        const c = v * s;
        const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
        const m = v - c;
        let r, g, b;
        if (h < 60) { r = c; g = x; b = 0; }
        else if (h < 120) { r = x; g = c; b = 0; }
        else if (h < 180) { r = 0; g = c; b = x; }
        else if (h < 240) { r = 0; g = x; b = c; }
        else if (h < 300) { r = x; g = 0; b = c; }
        else { r = c; g = 0; b = x; }
        return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
    }

    function getNetColors(numNets) {
        const colors = {};
        for (let i = 1; i <= numNets; i++) {
            const hue = (i / numNets) * 360;
            const [r, g, b] = hsvToRgb(hue, 1, 1);
            colors[i] = `rgb(${r},${g},${b})`;
        }
        return colors;
    }

    // --- Simple Union-Find for net grouping ---
    function computeNets(connections, pinAnchors) {
        const pinKeys = pinAnchors.map(p => `${p.comp_idx}_${p.pin_id}`);
        const parent = {};
        const rank = {};
        pinKeys.forEach(k => { parent[k] = k; rank[k] = 0; });

        function find(x) {
            if (!parent[x]) { parent[x] = x; rank[x] = 0; }
            if (parent[x] !== x) parent[x] = find(parent[x]);
            return parent[x];
        }
        function union(a, b) {
            const ra = find(a), rb = find(b);
            if (ra === rb) return;
            if (rank[ra] < rank[rb]) parent[ra] = rb;
            else if (rank[ra] > rank[rb]) parent[rb] = ra;
            else { parent[rb] = ra; rank[ra]++; }
        }

        connections.forEach(conn => {
            const k1 = `${conn.pin1.comp_idx}_${conn.pin1.pin_id}`;
            const k2 = `${conn.pin2.comp_idx}_${conn.pin2.pin_id}`;
            union(k1, k2);
        });

        // Assign net labels
        const rootToNet = {};
        let netCount = 0;
        const pinNetMap = {};
        pinKeys.forEach(k => {
            const root = find(k);
            if (!(root in rootToNet)) {
                netCount++;
                rootToNet[root] = netCount;
            }
            pinNetMap[k] = rootToNet[root];
        });

        // Only count nets that actually have connections (>= 2 pins)
        const netPinCounts = {};
        Object.values(pinNetMap).forEach(n => { netPinCounts[n] = (netPinCounts[n] || 0) + 1; });
        const connectedNets = Object.values(netPinCounts).filter(c => c >= 2).length;

        return { pinNetMap, netCount, connectedNets, netColors: getNetColors(netCount) };
    }

    // --- Pin lookup helper ---
    function findPinAnchor(comp_idx, pin_id) {
        return aiPreview.pinAnchors.find(p => p.comp_idx === comp_idx && p.pin_id === pin_id);
    }

    // --- Transform helpers for the preview canvas ---
    function aiScreenToImage(sx, sy) {
        return {
            x: (sx - aiEditor.pan.x) / aiEditor.zoom,
            y: (sy - aiEditor.pan.y) / aiEditor.zoom
        };
    }

    // --- Hit testing ---
    const AI_PIN_HIT_RADIUS = 50; // pixels in image space (generous for easy clicking)
    const AI_CONN_HIT_DIST = 35;  // pixels in image space for wire deletion

    function aiHitTestPin(imgX, imgY) {
        let best = null, bestDist = AI_PIN_HIT_RADIUS;
        for (const pin of aiPreview.pinAnchors) {
            const dx = imgX - pin.x;
            const dy = imgY - pin.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < bestDist) {
                bestDist = dist;
                best = pin;
            }
        }
        return best;
    }

    function pointToSegmentDist(px, py, ax, ay, bx, by) {
        const dx = bx - ax, dy = by - ay;
        const lenSq = dx * dx + dy * dy;
        if (lenSq === 0) return Math.sqrt((px - ax) ** 2 + (py - ay) ** 2);
        let t = ((px - ax) * dx + (py - ay) * dy) / lenSq;
        t = Math.max(0, Math.min(1, t));
        const projX = ax + t * dx, projY = ay + t * dy;
        return Math.sqrt((px - projX) ** 2 + (py - projY) ** 2);
    }

    function aiHitTestConnection(imgX, imgY) {
        let bestIdx = null, bestDist = AI_CONN_HIT_DIST;
        aiPreview.connections.forEach((conn, idx) => {
            const p1 = findPinAnchor(conn.pin1.comp_idx, conn.pin1.pin_id);
            const p2 = findPinAnchor(conn.pin2.comp_idx, conn.pin2.pin_id);
            if (!p1 || !p2) return;
            const dist = pointToSegmentDist(imgX, imgY, p1.x, p1.y, p2.x, p2.y);
            if (dist < bestDist) {
                bestDist = dist;
                bestIdx = idx;
            }
        });
        return bestIdx;
    }

    // --- Update stats badge ---
    function updateAiStats() {
        const nets = computeNets(aiPreview.connections, aiPreview.pinAnchors);
        aiStatsEl.innerHTML = `
            <span class="stat-item"><span class="stat-icon">📦</span> ${aiPreview.components.length} components</span>
            <span class="stat-item"><span class="stat-icon">🔗</span> ${aiPreview.connections.length} connections</span>
            <span class="stat-item"><span class="stat-icon">🌐</span> ${nets.connectedNets} nets</span>
            <span class="stat-item"><span class="stat-icon">📌</span> ${aiPreview.pinAnchors.length} pins</span>
        `;
    }

    // --- Main render for the preview canvas ---
    function renderAiPreview() {
        const cw = aiCanvas.width;
        const ch = aiCanvas.height;
        aiCtx.clearRect(0, 0, cw, ch);

        // Dark background
        aiCtx.fillStyle = '#080a10';
        aiCtx.fillRect(0, 0, cw, ch);

        aiCtx.save();
        aiCtx.translate(aiEditor.pan.x, aiEditor.pan.y);
        aiCtx.scale(aiEditor.zoom, aiEditor.zoom);

        // 1) Draw the background image
        if (aiPreview.imageElement) {
            aiCtx.drawImage(aiPreview.imageElement, 0, 0);
        }

        // Compute nets for coloring
        const nets = computeNets(aiPreview.connections, aiPreview.pinAnchors);

        // 2) Draw component bboxes
        aiCtx.lineWidth = 2 / aiEditor.zoom;
        aiPreview.components.forEach(comp => {
            const box = comp.box;
            if (!box || box.length < 4) return;
            
            const isHovered = aiEditor.hoveredComp && aiEditor.hoveredComp.name === comp.name;
            
            if (isHovered) {
                aiCtx.strokeStyle = '#10B981'; // vibrant green highlight
                aiCtx.fillStyle = 'rgba(16, 185, 129, 0.15)'; // light green fill
                aiCtx.fillRect(box[0], box[1], box[2], box[3]);
            } else {
                aiCtx.strokeStyle = 'rgba(255,255,255,0.5)';
            }
            
            aiCtx.strokeRect(box[0], box[1], box[2], box[3]);
            
            // Component label & value
            const fontSize = Math.max(11, 14 / aiEditor.zoom);
            aiCtx.font = `bold ${fontSize}px Inter, sans-serif`;
            aiCtx.fillStyle = isHovered ? '#10B981' : 'rgba(255,255,255,0.85)';
            
            const labelText = comp.name || comp.type;
            const valueText = comp.value ? ` (${comp.value})` : '';
            aiCtx.fillText(labelText + valueText, box[0] + 3, box[1] - 5);
        });

        // 3) Draw connection lines
        const connLineWidth = Math.max(2.5, 3 / aiEditor.zoom);
        aiPreview.connections.forEach((conn, idx) => {
            const p1 = findPinAnchor(conn.pin1.comp_idx, conn.pin1.pin_id);
            const p2 = findPinAnchor(conn.pin2.comp_idx, conn.pin2.pin_id);
            if (!p1 || !p2) return;

            const pinKey1 = `${conn.pin1.comp_idx}_${conn.pin1.pin_id}`;
            const netLabel = nets.pinNetMap[pinKey1];
            let color = nets.netColors[netLabel] || 'rgba(100,100,100,0.7)';

            const isHovered = aiEditor.hoveredConnIdx === idx;

            aiCtx.beginPath();
            aiCtx.moveTo(p1.x, p1.y);
            aiCtx.lineTo(p2.x, p2.y);
            aiCtx.lineWidth = isHovered ? connLineWidth * 2 : connLineWidth;
            aiCtx.strokeStyle = isHovered ? '#EF4444' : color;
            if (isHovered) {
                aiCtx.setLineDash([6 / aiEditor.zoom, 4 / aiEditor.zoom]);
            }
            aiCtx.stroke();
            aiCtx.setLineDash([]);

            // Connection midpoint marker when hovered (delete mode)
            if (isHovered && aiEditor.tool === 'delete') {
                const mx = (p1.x + p2.x) / 2, my = (p1.y + p2.y) / 2;
                const r = 10 / aiEditor.zoom;
                aiCtx.beginPath();
                aiCtx.arc(mx, my, r, 0, Math.PI * 2);
                aiCtx.fillStyle = 'rgba(239, 68, 68, 0.85)';
                aiCtx.fill();
                aiCtx.strokeStyle = 'white';
                aiCtx.lineWidth = 1.5 / aiEditor.zoom;
                aiCtx.stroke();
                // "X" mark
                const cross = r * 0.5;
                aiCtx.beginPath();
                aiCtx.moveTo(mx - cross, my - cross); aiCtx.lineTo(mx + cross, my + cross);
                aiCtx.moveTo(mx + cross, my - cross); aiCtx.lineTo(mx - cross, my + cross);
                aiCtx.strokeStyle = 'white';
                aiCtx.lineWidth = 2 / aiEditor.zoom;
                aiCtx.stroke();
            }
        });

        // 4) Draw pin dots (large, interactive)
        const pinRadius = Math.max(8, 10 / aiEditor.zoom);
        const pinFontSize = Math.max(9, 11 / aiEditor.zoom);
        aiPreview.pinAnchors.forEach(pin => {
            const pinKey = `${pin.comp_idx}_${pin.pin_id}`;
            const netLabel = nets.pinNetMap[pinKey];
            let fillColor = nets.netColors[netLabel] || 'rgba(128,128,128,0.8)';
            const isJunction = pin.isJunction || pin.comp_idx === -1;

            // Check if this pin is connected to anything
            const isConnected = aiPreview.connections.some(c =>
                (c.pin1.comp_idx === pin.comp_idx && c.pin1.pin_id === pin.pin_id) ||
                (c.pin2.comp_idx === pin.comp_idx && c.pin2.pin_id === pin.pin_id)
            );

            const isHovered = aiEditor.hoveredPin &&
                aiEditor.hoveredPin.comp_idx === pin.comp_idx &&
                aiEditor.hoveredPin.pin_id === pin.pin_id;
            const isPending = aiEditor.pendingPin &&
                aiEditor.pendingPin.comp_idx === pin.comp_idx &&
                aiEditor.pendingPin.pin_id === pin.pin_id;

            const isHoveredJunc = aiEditor.hoveredJunction &&
                aiEditor.hoveredJunction.comp_idx === pin.comp_idx &&
                aiEditor.hoveredJunction.pin_id === pin.pin_id;

            // Draw outer glow for hovered/pending/hoveredJunc
            if (isHovered || isPending || isHoveredJunc) {
                aiCtx.beginPath();
                aiCtx.arc(pin.x, pin.y, pinRadius * 1.8, 0, Math.PI * 2);
                aiCtx.fillStyle = isPending ? 'rgba(0, 229, 255, 0.25)' : (isHoveredJunc ? 'rgba(239, 68, 68, 0.25)' : 'rgba(255,255,255,0.15)');
                aiCtx.fill();
            }

            if (isJunction) {
                // Junction: draw as a diamond ◇
                const r = pinRadius * 1.2;
                aiCtx.beginPath();
                aiCtx.moveTo(pin.x, pin.y - r);
                aiCtx.lineTo(pin.x + r, pin.y);
                aiCtx.lineTo(pin.x, pin.y + r);
                aiCtx.lineTo(pin.x - r, pin.y);
                aiCtx.closePath();

                if (isHoveredJunc) {
                    aiCtx.fillStyle = 'rgba(239, 68, 68, 0.9)'; // Red delete highlight
                    aiCtx.strokeStyle = '#EF4444';
                } else {
                    aiCtx.fillStyle = isPending ? '#00E5FF' : (isConnected ? fillColor : 'rgba(255, 200, 50, 0.7)');
                    aiCtx.strokeStyle = isHovered ? '#FFF' : 'rgba(255,255,255,0.8)';
                }
                aiCtx.fill();
                aiCtx.lineWidth = (isHovered || isHoveredJunc ? 2.5 : 1.5) / aiEditor.zoom;
                aiCtx.stroke();

                // "+" cross inside
                const cr = r * 0.45;
                aiCtx.beginPath();
                aiCtx.moveTo(pin.x - cr, pin.y); aiCtx.lineTo(pin.x + cr, pin.y);
                aiCtx.moveTo(pin.x, pin.y - cr); aiCtx.lineTo(pin.x, pin.y + cr);
                aiCtx.strokeStyle = 'rgba(255,255,255,0.9)';
                aiCtx.lineWidth = 2 / aiEditor.zoom;
                aiCtx.stroke();

                // Label: J#
                aiCtx.font = `bold ${pinFontSize}px Inter, sans-serif`;
                aiCtx.fillStyle = 'rgba(255,200,50,0.95)';
                aiCtx.textBaseline = 'middle';
                const labelX = pin.x + r + 4 / aiEditor.zoom;
                aiCtx.strokeStyle = 'rgba(0,0,0,0.7)';
                aiCtx.lineWidth = 3 / aiEditor.zoom;
                aiCtx.strokeText(`J${pin.pin_id}`, labelX, pin.y);
                aiCtx.fillText(`J${pin.pin_id}`, labelX, pin.y);
            } else {
                // Regular pin: circle
                aiCtx.beginPath();
                aiCtx.arc(pin.x, pin.y, pinRadius, 0, Math.PI * 2);
                aiCtx.fillStyle = isPending ? '#00E5FF' : (isConnected ? fillColor : 'rgba(100,100,100,0.6)');
                aiCtx.fill();
                aiCtx.strokeStyle = isHovered ? '#FFF' : 'rgba(255,255,255,0.7)';
                aiCtx.lineWidth = (isHovered ? 2.5 : 1.5) / aiEditor.zoom;
                aiCtx.stroke();

                // Pin label
                aiCtx.font = `bold ${pinFontSize}px Inter, sans-serif`;
                aiCtx.fillStyle = 'rgba(255,255,255,0.95)';
                aiCtx.textBaseline = 'middle';
                const labelX = pin.x + pinRadius + 4 / aiEditor.zoom;
                // Shadow for readability
                aiCtx.strokeStyle = 'rgba(0,0,0,0.7)';
                aiCtx.lineWidth = 3 / aiEditor.zoom;
                aiCtx.strokeText(`${pin.comp_idx}_${pin.pin_id}`, labelX, pin.y);
                aiCtx.fillText(`${pin.comp_idx}_${pin.pin_id}`, labelX, pin.y);
            }
        });

        // 5) Draw pending wire (from pendingPin to cursor)
        if (aiEditor.pendingPin && aiEditor.tool === 'connect') {
            const srcPin = findPinAnchor(aiEditor.pendingPin.comp_idx, aiEditor.pendingPin.pin_id);
            if (srcPin) {
                const cursorImg = aiScreenToImage(aiEditor.mouseX, aiEditor.mouseY);
                aiCtx.beginPath();
                aiCtx.moveTo(srcPin.x, srcPin.y);
                aiCtx.lineTo(cursorImg.x, cursorImg.y);
                aiCtx.strokeStyle = 'rgba(0, 229, 255, 0.7)';
                aiCtx.lineWidth = 2.5 / aiEditor.zoom;
                aiCtx.setLineDash([8 / aiEditor.zoom, 4 / aiEditor.zoom]);
                aiCtx.stroke();
                aiCtx.setLineDash([]);
            }
        }

        aiCtx.restore();
    }

    // --- Resize the preview canvas ---
    function resizeAiCanvas() {
        const rect = aiCanvasWrap.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        aiCanvas.width = rect.width * dpr;
        aiCanvas.height = rect.height * dpr;
        aiCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
        // Store display size for coordinate conversion
        aiCanvas._displayWidth = rect.width;
        aiCanvas._displayHeight = rect.height;
    }

    // --- Fit the image in the canvas ---
    function fitAiPreview() {
        if (!aiPreview.imageElement) return;
        const cw = aiCanvas._displayWidth || aiCanvasWrap.clientWidth;
        const ch = aiCanvas._displayHeight || aiCanvasWrap.clientHeight;
        const iw = aiPreview.imageW;
        const ih = aiPreview.imageH;
        const margin = 40;
        const scaleX = (cw - margin * 2) / iw;
        const scaleY = (ch - margin * 2) / ih;
        aiEditor.zoom = Math.min(scaleX, scaleY, 2.0);
        aiEditor.pan.x = (cw - iw * aiEditor.zoom) / 2;
        aiEditor.pan.y = (ch - ih * aiEditor.zoom) / 2;
    }

    // --- Set tool and update UI ---
    function setAiTool(toolName) {
        aiEditor.tool = toolName;
        aiEditor.pendingPin = null;
        aiEditor.hoveredPin = null;
        aiEditor.hoveredConnIdx = null;
        aiEditor.hoveredComp = null;
        if (aiCanvas) aiCanvas.style.cursor = '';

        // Update button states
        document.querySelectorAll('.ai-debug-tool').forEach(btn => btn.classList.remove('active'));
        if (toolName === 'select') document.getElementById('aiToolSelect').classList.add('active');
        if (toolName === 'connect') document.getElementById('aiToolWire').classList.add('active');
        if (toolName === 'delete') document.getElementById('aiToolDelete').classList.add('active');
        if (toolName === 'junction') document.getElementById('aiToolJunction').classList.add('active');

        // Update cursor class
        aiCanvasWrap.className = 'ai-debug-canvas-wrap tool-' + toolName;

        // Update info text
        const messages = {
            select: 'Scroll to zoom · Drag to pan (works in all modes via middle-click)',
            connect: 'Click a pin to start, click another to connect · Middle-click to pan',
            delete: 'Click on a connection line to remove it · Middle-click to pan',
            junction: 'Click to place a junction pin · Then use Connect to wire it'
        };
        aiInfoEl.textContent = messages[toolName] || '';
        aiInfoEl.style.color = '';

        renderAiPreview();
    }

    // --- Next junction ID counter ---
    let aiJunctionCounter = 0;

    // --- Tool button listeners ---
    document.getElementById('aiToolSelect').addEventListener('click', () => setAiTool('select'));
    document.getElementById('aiToolWire').addEventListener('click', () => setAiTool('connect'));
    document.getElementById('aiToolDelete').addEventListener('click', () => setAiTool('delete'));
    document.getElementById('aiToolJunction').addEventListener('click', () => setAiTool('junction'));

    // --- Canvas mouse events ---
    aiCanvas.addEventListener('mousedown', (e) => {
        const rect = aiCanvas.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;
        const img = aiScreenToImage(sx, sy);

        aiEditor.isMouseDown = true;
        aiEditor.dragDistance = 0;
        aiEditor.dragStart = { x: e.clientX, y: e.clientY };
        aiEditor.panStart = { ...aiEditor.pan };

        // Middle-click (1) or right-click (2) starts panning immediately
        if (e.button === 1 || e.button === 2) {
            e.preventDefault();
            aiEditor.isDragging = true;
            aiCanvasWrap.classList.add('dragging');
            return;
        }

        if (e.button === 0) { // Left-click
            // Check if they clicked on a virtual junction pin
            const clickedPin = aiHitTestPin(img.x, img.y);
            const isJunc = clickedPin && clickedPin.comp_idx === -1;

            if (isJunc && (aiEditor.tool === 'select' || aiEditor.tool === 'junction')) {
                // Drag the junction to move it
                aiEditor.isDraggingJunction = true;
                aiEditor.draggedJunction = clickedPin;
                return;
            }

            if (isJunc && aiEditor.tool === 'delete') {
                // Delete the junction and all connections connected to it
                const pinId = clickedPin.pin_id;
                aiPreview.connections = aiPreview.connections.filter(c =>
                    !(c.pin1.comp_idx === -1 && c.pin1.pin_id === pinId) &&
                    !(c.pin2.comp_idx === -1 && c.pin2.pin_id === pinId)
                );
                aiPreview.pinAnchors = aiPreview.pinAnchors.filter(a =>
                    !(a.comp_idx === -1 && a.pin_id === pinId)
                );
                aiInfoEl.textContent = `🗑️ Removed junction J${pinId} and its connections.`;
                aiInfoEl.style.color = '#EF4444';
                aiEditor.hoveredJunction = null;
                updateAiStats();
                renderAiPreview();
                return;
            }

            if (aiEditor.tool === 'select') {
                // Left-click in select mode starts panning immediately
                aiEditor.isDragging = true;
                aiCanvasWrap.classList.add('dragging');
            } else if (aiEditor.tool === 'connect') {
                const pin = clickedPin;
                if (pin) {
                    if (!aiEditor.pendingPin) {
                        // First pin
                        aiEditor.pendingPin = { comp_idx: pin.comp_idx, pin_id: pin.pin_id };
                        aiInfoEl.textContent = `Pin ${pin.comp_idx}_${pin.pin_id} selected — click another pin to connect`;
                        aiInfoEl.style.color = '#00E5FF';
                    } else {
                        // Second pin — create connection
                        const p1 = aiEditor.pendingPin;
                        const p2 = { comp_idx: pin.comp_idx, pin_id: pin.pin_id };

                        // Don't connect a pin to itself
                        if (p1.comp_idx === p2.comp_idx && p1.pin_id === p2.pin_id) {
                            aiEditor.pendingPin = null;
                            aiInfoEl.textContent = 'Same pin — cancelled. Click a pin to start.';
                            aiInfoEl.style.color = '';
                        } else {
                            // Check if connection already exists
                            const exists = aiPreview.connections.some(c =>
                                (c.pin1.comp_idx === p1.comp_idx && c.pin1.pin_id === p1.pin_id &&
                                    c.pin2.comp_idx === p2.comp_idx && c.pin2.pin_id === p2.pin_id) ||
                                (c.pin1.comp_idx === p2.comp_idx && c.pin1.pin_id === p2.pin_id &&
                                    c.pin2.comp_idx === p1.comp_idx && c.pin2.pin_id === p1.pin_id)
                            );
                            if (exists) {
                                aiInfoEl.textContent = 'Connection already exists! Click a pin to start.';
                                aiInfoEl.style.color = '#EF4444';
                            } else {
                                aiPreview.connections.push({ pin1: p1, pin2: p2 });
                                aiInfoEl.textContent = `✅ Connected ${p1.comp_idx}_${p1.pin_id} → ${p2.comp_idx}_${p2.pin_id}`;
                                aiInfoEl.style.color = '#10B981';
                                updateAiStats();
                            }
                            aiEditor.pendingPin = null;
                        }
                    }
                    renderAiPreview();
                }
            } else if (aiEditor.tool === 'delete') {
                if (aiEditor.hoveredConnIdx !== null) {
                    const removed = aiPreview.connections.splice(aiEditor.hoveredConnIdx, 1)[0];
                    aiInfoEl.textContent = `🗑️ Removed connection ${removed.pin1.comp_idx}_${removed.pin1.pin_id} — ${removed.pin2.comp_idx}_${removed.pin2.pin_id}`;
                    aiInfoEl.style.color = '#EF4444';
                    aiEditor.hoveredConnIdx = null;
                    updateAiStats();
                    renderAiPreview();
                }
            }
        }
    });

    // Prevent middle-click/right-click default behaviors
    aiCanvas.addEventListener('auxclick', (e) => { if (e.button === 1) e.preventDefault(); });

    aiCanvas.addEventListener('mousemove', (e) => {
        const rect = aiCanvas.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;
        aiEditor.mouseX = sx;
        aiEditor.mouseY = sy;
        const img = aiScreenToImage(sx, sy);

        if (!aiEditor.isMouseDown) {
            if (aiEditor.tool === 'select') {
                const hoveredComp = aiPreview.components.find(comp => {
                    const box = comp.box;
                    if (!box || box.length < 4) return false;
                    return img.x >= box[0] && img.x <= box[0] + box[2] &&
                           img.y >= box[1] && img.y <= box[1] + box[3];
                });
                if (hoveredComp) {
                    aiCanvas.style.cursor = 'pointer';
                    aiEditor.hoveredComp = hoveredComp;
                } else {
                    aiCanvas.style.cursor = '';
                    aiEditor.hoveredComp = null;
                }
            } else {
                aiCanvas.style.cursor = '';
                aiEditor.hoveredComp = null;
            }
        }

        if (aiEditor.isMouseDown) {
            const dx = e.clientX - aiEditor.dragStart.x;
            const dy = e.clientY - aiEditor.dragStart.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            aiEditor.dragDistance = dist;

            // Handle dragging a junction
            if (aiEditor.isDraggingJunction && aiEditor.draggedJunction) {
                const juncPin = aiPreview.pinAnchors.find(a => a.comp_idx === -1 && a.pin_id === aiEditor.draggedJunction.pin_id);
                if (juncPin) {
                    juncPin.x = Math.round(img.x);
                    juncPin.y = Math.round(img.y);
                }
                renderAiPreview();
                return;
            }

            // Start dragging (pan) if they moved more than 4 pixels
            if (!aiEditor.isDragging && dist > 4) {
                aiEditor.isDragging = true;
                aiCanvasWrap.classList.add('dragging');
            }

            if (aiEditor.isDragging) {
                aiEditor.pan.x = aiEditor.panStart.x + dx;
                aiEditor.pan.y = aiEditor.panStart.y + dy;
                renderAiPreview();
                return;
            }
        }

        // Highlight junctions when hovering over them in select/junction modes (to indicate draggable)
        if (aiEditor.tool === 'select' || aiEditor.tool === 'junction') {
            const pin = aiHitTestPin(img.x, img.y);
            if (pin && pin.comp_idx === -1) {
                aiEditor.hoveredPin = pin;
            } else {
                aiEditor.hoveredPin = null;
            }
            renderAiPreview();
        } else if (aiEditor.tool === 'connect') {
            const pin = aiHitTestPin(img.x, img.y);
            aiEditor.hoveredPin = pin;
            if (pin) {
                aiCanvasWrap.classList.add('pin-hover');
            } else {
                aiCanvasWrap.classList.remove('pin-hover');
            }
            renderAiPreview();
        } else if (aiEditor.tool === 'delete') {
            const pin = aiHitTestPin(img.x, img.y);
            if (pin && pin.comp_idx === -1) {
                aiEditor.hoveredJunction = pin;
                aiEditor.hoveredConnIdx = null;
            } else {
                aiEditor.hoveredJunction = null;
                aiEditor.hoveredConnIdx = aiHitTestConnection(img.x, img.y);
            }
            renderAiPreview();
        }
    });

    aiCanvas.addEventListener('mouseup', (e) => {
        const rect = aiCanvas.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;
        const img = aiScreenToImage(sx, sy);

        const wasDragging = aiEditor.isDragging;
        const wasDraggingJunc = aiEditor.isDraggingJunction;

        aiEditor.isMouseDown = false;
        aiEditor.isDragging = false;
        aiEditor.isDraggingJunction = false;
        aiEditor.draggedJunction = null;
        aiCanvasWrap.classList.remove('dragging');

        if (e.button === 0 && !wasDragging && !wasDraggingJunc && aiEditor.dragDistance < 5) {
            if (aiEditor.tool === 'select') {
                const clickedComp = aiPreview.components.find(comp => {
                    const box = comp.box;
                    if (!box || box.length < 4) return false;
                    return img.x >= box[0] && img.x <= box[0] + box[2] &&
                           img.y >= box[1] && img.y <= box[1] + box[3];
                });
                if (clickedComp) {
                    const oldValue = clickedComp.value === "TEXT_FOUND" ? "" : (clickedComp.value || "");
                    const newValue = prompt(`Edit value for component ${clickedComp.name} (${clickedComp.type}):`, oldValue);
                    if (newValue !== null) {
                        clickedComp.value = newValue;
                        aiInfoEl.textContent = `✍️ Updated ${clickedComp.name} value to ${newValue}`;
                        aiInfoEl.style.color = '#10B981';
                        renderAiPreview();
                    }
                    return;
                }
            } else if (aiEditor.tool === 'junction') {
                const juncId = aiJunctionCounter++;
                const juncCompIdx = -1; // virtual junction component
                const juncPinId = juncId;
                aiPreview.pinAnchors.push({
                    comp_idx: juncCompIdx,
                    pin_id: juncPinId,
                    x: Math.round(img.x),
                    y: Math.round(img.y),
                    isJunction: true
                });
                aiInfoEl.textContent = `⊕ Placed junction J${juncId} — use Connect to wire it`;
                aiInfoEl.style.color = '#10B981';
                updateAiStats();
                renderAiPreview();
            } else if (aiEditor.tool === 'connect') {
                const pin = aiHitTestPin(img.x, img.y);
                if (!pin && aiEditor.pendingPin) {
                    aiEditor.pendingPin = null;
                    aiInfoEl.textContent = 'Cancelled. Click a pin to start a connection.';
                    aiInfoEl.style.color = '';
                    renderAiPreview();
                }
            }
        }
    });

    aiCanvas.addEventListener('mouseleave', () => {
        if (aiEditor.isDragging) {
            aiEditor.isDragging = false;
            aiCanvasWrap.classList.remove('dragging');
        }
        aiEditor.hoveredPin = null;
        aiEditor.hoveredConnIdx = null;
        aiEditor.hoveredComp = null;
        aiCanvas.style.cursor = '';
        renderAiPreview();
    });

    // Zoom with scroll wheel
    aiCanvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        const rect = aiCanvas.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;

        const zoomFactor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
        const newZoom = Math.max(0.1, Math.min(10, aiEditor.zoom * zoomFactor));

        // Zoom towards cursor
        aiEditor.pan.x = sx - (sx - aiEditor.pan.x) * (newZoom / aiEditor.zoom);
        aiEditor.pan.y = sy - (sy - aiEditor.pan.y) * (newZoom / aiEditor.zoom);
        aiEditor.zoom = newZoom;

        renderAiPreview();
    }, { passive: false });

    // Right-click to cancel pending connection
    aiCanvas.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        if (aiEditor.pendingPin) {
            aiEditor.pendingPin = null;
            aiInfoEl.textContent = 'Cancelled. Click a pin to start.';
            aiInfoEl.style.color = '';
            renderAiPreview();
        }
    });

    // --- Initialize the preview editor with API data ---
    function initAiPreviewEditor(data) {
        // Reset junction counter
        aiJunctionCounter = 0;

        // Store data
        aiPreview.components = data.components || [];
        aiPreview.connections = JSON.parse(JSON.stringify(data.connections || [])); // deep copy for editing
        aiPreview.pinAnchors = (data.pin_anchors || []).map(p => ({
            comp_idx: p.comp_idx,
            pin_id: p.pin_id,
            x: p.x,
            y: p.y
        }));

        // Reset editor state
        aiEditor.tool = 'select';
        aiEditor.pendingPin = null;
        aiEditor.hoveredPin = null;
        aiEditor.hoveredConnIdx = null;
        aiEditor.hoveredComp = null;
        aiCanvas.style.cursor = '';
        aiEditor.isDragging = false;

        // Load original image from debug_image or uploaded file
        const img = new Image();
        img.onload = () => {
            aiPreview.imageElement = img;
            aiPreview.imageW = img.naturalWidth;
            aiPreview.imageH = img.naturalHeight;

            resizeAiCanvas();
            fitAiPreview();
            setAiTool('select');
            updateAiStats();
            renderAiPreview();
        };

        if (data.debug_image) {
            img.src = data.debug_image;
        }
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
                document.getElementById("statusText").innerText = `✅ Loaded ${data.components.length} components. Review & edit connections.`;

                // Show modal and initialize the interactive preview editor
                const modal = document.getElementById("aiDebugModal");
                modal.style.display = "flex";

                // Initialize the interactive editor with the API data
                // Use requestAnimationFrame to ensure the modal is visible before sizing canvas
                requestAnimationFrame(() => {
                    initAiPreviewEditor(data);
                });

                // Setup import logic for when user clicks "Import to Canvas"
                const btnImport = document.getElementById("btnAiDebugImport");
                // Remove any old listeners by cloning (simple way)
                const newBtnImport = btnImport.cloneNode(true);
                btnImport.parentNode.replaceChild(newBtnImport, btnImport);

                newBtnImport.addEventListener("click", () => {
                    saveState();
                    modal.style.display = "none";

                    // Use the EDITED connections from the preview editor
                    const editedConnections = aiPreview.connections;

                    document.getElementById("statusText").innerText = `✅ Imported ${data.components.length} components with ${editedConnections.length} connections.`;

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
                                nodes: c.nodes || [],
                                rotation: c.rotation || 0
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

                    // Load Wires using the EDITED connections
                    wires = [];
                    if (editedConnections && editedConnections.length > 0) {
                        editedConnections.forEach(conn => {
                            // Helper to resolve pin to main editor coordinate space
                            const getPinCoords = (pinRef) => {
                                if (pinRef.comp_idx === -1) {
                                    // It's a junction. Find the corresponding anchor in aiPreview.pinAnchors
                                    const anchor = aiPreview.pinAnchors.find(a => a.comp_idx === -1 && a.pin_id === pinRef.pin_id);
                                    if (anchor) {
                                        return {
                                            x: snap(anchor.x * SCALE_FACTOR),
                                            y: snap(anchor.y * SCALE_FACTOR)
                                        };
                                    }
                                    return null;
                                } else {
                                    // It's a regular component pin
                                    const idx = compIdxMap[pinRef.comp_idx];
                                    if (idx !== undefined && idx !== -1) {
                                        const comp = components[idx];
                                        const pins = getCompPins(comp);
                                        return pins[pinRef.pin_id];
                                    }
                                    return null;
                                }
                            };

                            const p1 = getPinCoords(conn.pin1);
                            const p2 = getPinCoords(conn.pin2);

                            if (p1 && p2 && (p1.x !== p2.x || p1.y !== p2.y)) {
                                const comp1 = components[compIdxMap[conn.pin1.comp_idx]];
                                const comp2 = components[compIdxMap[conn.pin2.comp_idx]];
                                const exclude = [comp1, comp2].filter(Boolean);

                                if (p1.x === p2.x || p1.y === p2.y) {
                                    const routed = routeAroundComponent(p1, p2, components, exclude);
                                    routed.forEach(rSeg => wires.push(rSeg));
                                } else {
                                    // Collision-aware L-shaped routing
                                    const mid1 = { x: p2.x, y: p1.y };
                                    const mid2 = { x: p1.x, y: p2.y };
                                    
                                    const coll1 = doesSegmentIntersectComponent(p1, mid1, components, exclude) || 
                                                  doesSegmentIntersectComponent(mid1, p2, components, exclude);
                                    const coll2 = doesSegmentIntersectComponent(p1, mid2, components, exclude) || 
                                                  doesSegmentIntersectComponent(mid2, p2, components, exclude);
                                    
                                    let chosenSegments = [];
                                    if (coll1 && !coll2) {
                                        // Option 1 (H-then-V) collides, but Option 2 (V-then-H) is clean. Choose V-then-H
                                        chosenSegments = [
                                            [p1, { x: p1.x, y: p2.y }],
                                            [{ x: p1.x, y: p2.y }, p2]
                                        ];
                                    } else {
                                        // Default to Option 1
                                        chosenSegments = [
                                            [p1, { x: p2.x, y: p1.y }],
                                            [{ x: p2.x, y: p1.y }, p2]
                                        ];
                                    }

                                    chosenSegments.forEach(seg => {
                                        if (seg[0].x !== seg[1].x || seg[0].y !== seg[1].y) {
                                            const routed = routeAroundComponent(seg[0], seg[1], components, exclude);
                                            routed.forEach(rSeg => wires.push(rSeg));
                                        }
                                    });
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
                    selectedComponents = [];
                    selectedComp = null;
                    selectedWires = [];
                    selectedWirePts = [];

                    // Auto-fix any wires that run through component bodies after import.
                    // Overlapping component placements from AI detection can cause wires to
                    // intersect component hitboxes — rerouting resolves this silently.
                    rerouteAllWires();

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

    // Handle window resize for the AI preview canvas
    window.addEventListener('resize', () => {
        if (document.getElementById("aiDebugModal").style.display !== "none") {
            resizeAiCanvas();
            renderAiPreview();
        }
    });

    function getNodeAt(pt) {
        if (!nodeMap) return null;
        return nodeMap[`${snap(pt.x)},${snap(pt.y)}`];
    }

    let animationFrameId = null;
    function scheduleAnimation() {
        const hasHighlights = highlightedComponents.length > 0 || highlightedNodes.length > 0;
        if (hasHighlights) {
            if (!animationFrameId) {
                const anim = () => {
                    render();
                    animationFrameId = requestAnimationFrame(anim);
                };
                animationFrameId = requestAnimationFrame(anim);
            }
        } else {
            if (animationFrameId) {
                cancelAnimationFrame(animationFrameId);
                animationFrameId = null;
            }
        }
    }

    function updateNetlistPreview(netlistText, highlightLineNum = null) {
        const netlistEl = document.getElementById('netlistText');
        if (!netlistEl) return;
        
        if (netlistText !== undefined) {
            lastNetlistText = netlistText;
        } else {
            netlistText = lastNetlistText;
        }

        if (!netlistText) {
            netlistEl.innerHTML = '<span class="placeholder-text">Click RUN to generate</span>';
            return;
        }
        
        const lines = netlistText.split('\n');
        netlistEl.innerHTML = '';
        
        lines.forEach((line, index) => {
            const lineNum = index + 1;
            const lineRow = document.createElement('div');
            lineRow.className = 'netlist-line-row';
            if (lineNum === highlightLineNum) {
                lineRow.classList.add('netlist-line-highlighted');
            }
            lineRow.id = `netlist-line-${lineNum}`;
            
            const lineNumSp = document.createElement('span');
            lineNumSp.className = 'netlist-line-number';
            lineNumSp.textContent = lineNum.toString().padStart(3, ' ');
            
            const lineTextSp = document.createElement('span');
            lineTextSp.className = 'netlist-line-content';
            lineTextSp.textContent = line;
            
            lineRow.appendChild(lineNumSp);
            lineRow.appendChild(lineTextSp);
            netlistEl.appendChild(lineRow);
        });

        if (highlightLineNum !== null) {
            const targetRow = document.getElementById(`netlist-line-${highlightLineNum}`);
            if (targetRow) {
                targetRow.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        }
    }

    function locateComponent(name) {
        const comp = components.find(c => c.name === name);
        if (!comp) return;
        
        selectedComponents = [comp];
        selectedComp = comp;
        updatePropertiesPanel();

        offsetX = canvas.width / 2 - comp.x * zoom;
        offsetY = canvas.height / 2 - comp.y * zoom;

        highlightedComponents = [comp];
        highlightedNodes = [];
        highlightedNetlistLine = null;
        scheduleAnimation();
        render();
    }

    function locateNode(nodeId) {
        const pts = [];
        for (const [key, node] of Object.entries(nodeMap)) {
            if (node === nodeId) {
                const [x, y] = key.split(',').map(Number);
                pts.push({ x, y });
            }
        }
        
        if (pts.length === 0) return;

        const avgX = pts.reduce((sum, p) => sum + p.x, 0) / pts.length;
        const avgY = pts.reduce((sum, p) => sum + p.y, 0) / pts.length;
        offsetX = canvas.width / 2 - avgX * zoom;
        offsetY = canvas.height / 2 - avgY * zoom;

        highlightedComponents = [];
        highlightedNodes = [nodeId];
        highlightedNetlistLine = null;
        scheduleAnimation();
        render();
    }

    function locateNetlistLine(lineNum) {
        highlightedNetlistLine = lineNum;
        updateNetlistPreview(undefined, lineNum);
        
        const panel = document.getElementById("netlistPreview");
        if (panel && panel.classList.contains('collapsed')) {
            panel.classList.remove('collapsed');
            const netlistToggle = document.getElementById("netlistToggle");
            if (netlistToggle) {
                const icon = netlistToggle.querySelector('.toggle-icon');
                if (icon) icon.textContent = '▼';
            }
        }
    }

    // ═══════════════════════════════════════════
    // CONSOLE LOG DRAWER
    // ═══════════════════════════════════════════
    const consoleDrawer = document.getElementById('consoleDrawer');
    const consoleHeader = document.getElementById('consoleHeader');
    const consoleBody = document.getElementById('consoleBody');
    const consoleStats = document.getElementById('consoleStats');
    
    const consoleTabAll = document.getElementById('consoleTabAll');
    const consoleTabErrors = document.getElementById('consoleTabErrors');
    const consoleTabWarnings = document.getElementById('consoleTabWarnings');
    const consoleBadgeErrors = document.getElementById('consoleBadgeErrors');
    const consoleBadgeWarnings = document.getElementById('consoleBadgeWarnings');
    
    const consoleBtnClear = document.getElementById('consoleBtnClear');
    const consoleBtnToggle = document.getElementById('consoleBtnToggle');

    function toggleConsole(expand) {
        if (!consoleDrawer) return;
        if (expand === undefined) {
            consoleDrawer.classList.toggle('expanded');
        } else if (expand) {
            consoleDrawer.classList.add('expanded');
        } else {
            consoleDrawer.classList.remove('expanded');
        }
        if (consoleBtnToggle) {
            consoleBtnToggle.textContent = consoleDrawer.classList.contains('expanded') ? '▼' : '▲';
        }
    }

    if (consoleHeader) {
        consoleHeader.addEventListener('click', (e) => {
            if (e.target.closest('.console-tab-btn') || e.target.closest('.console-btn-action')) {
                return;
            }
            toggleConsole();
        });
    }
    
    if (consoleBtnToggle) {
        consoleBtnToggle.addEventListener('click', () => {
            toggleConsole();
        });
    }

    if (consoleBtnClear) {
        consoleBtnClear.addEventListener('click', (e) => {
            e.stopPropagation();
            clearConsole();
        });
    }

    function clearConsole() {
        consoleLogs = [];
        highlightedComponents = [];
        highlightedNodes = [];
        highlightedNetlistLine = null;
        scheduleAnimation();
        updateNetlistPreview(lastNetlistText);
        renderConsole();
        render();
    }

    if (consoleTabAll) {
        consoleTabAll.addEventListener('click', (e) => {
            e.stopPropagation();
            setConsoleFilter('all');
        });
    }
    if (consoleTabErrors) {
        consoleTabErrors.addEventListener('click', (e) => {
            e.stopPropagation();
            setConsoleFilter('error');
        });
    }
    if (consoleTabWarnings) {
        consoleTabWarnings.addEventListener('click', (e) => {
            e.stopPropagation();
            setConsoleFilter('warning');
        });
    }

    function setConsoleFilter(filter) {
        consoleFilter = filter;
        [consoleTabAll, consoleTabErrors, consoleTabWarnings].forEach(btn => {
            if (btn) btn.classList.remove('active');
        });
        if (filter === 'all' && consoleTabAll) consoleTabAll.classList.add('active');
        if (filter === 'error' && consoleTabErrors) consoleTabErrors.classList.add('active');
        if (filter === 'warning' && consoleTabWarnings) consoleTabWarnings.classList.add('active');
        renderConsole();
    }

    function updateConsoleStats() {
        const errorsCount = consoleLogs.filter(log => log.type === 'error').length;
        const warningsCount = consoleLogs.filter(log => log.type === 'warning').length;

        if (consoleStats) {
            if (consoleLogs.length === 0) {
                consoleStats.textContent = '';
            } else {
                consoleStats.textContent = `(${errorsCount} Error${errorsCount !== 1 ? 's' : ''}, ${warningsCount} Warning${warningsCount !== 1 ? 's' : ''})`;
            }
        }

        if (consoleBadgeErrors) {
            if (errorsCount > 0) {
                consoleBadgeErrors.style.display = 'inline-block';
                consoleBadgeErrors.textContent = errorsCount;
            } else {
                consoleBadgeErrors.style.display = 'none';
            }
        }
        if (consoleBadgeWarnings) {
            if (warningsCount > 0) {
                consoleBadgeWarnings.style.display = 'inline-block';
                consoleBadgeWarnings.textContent = warningsCount;
            } else {
                consoleBadgeWarnings.style.display = 'none';
            }
        }
    }

    function renderConsole() {
        if (!consoleBody) return;
        
        updateConsoleStats();

        const filteredLogs = consoleLogs.filter(log => {
            if (consoleFilter === 'all') return true;
            return log.type === consoleFilter;
        });

        if (filteredLogs.length === 0) {
            consoleBody.innerHTML = `
                <div class="console-placeholder">
                    ${consoleFilter === 'all' 
                        ? 'No output. Run a simulation or solve nodes to inspect results.' 
                        : `No ${consoleFilter}s found.`}
                </div>
            `;
            return;
        }

        consoleBody.innerHTML = '';
        filteredLogs.forEach(log => {
            const row = document.createElement('div');
            row.className = `console-log-row log-${log.type}`;

            const icon = document.createElement('span');
            icon.className = 'log-icon';
            icon.textContent = log.type === 'error' ? '❌' : (log.type === 'warning' ? '⚠️' : 'ℹ️');

            const source = document.createElement('span');
            source.className = 'log-source';
            source.textContent = log.source;

            const msg = document.createElement('span');
            msg.className = 'log-message';
            msg.textContent = log.message;

            row.appendChild(icon);
            row.appendChild(source);
            row.appendChild(msg);

            const actions = document.createElement('div');
            actions.className = 'log-actions';

            if (log.component) {
                const btn = document.createElement('button');
                btn.className = 'log-action-btn';
                btn.textContent = `🔍 Locate ${log.component}`;
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    locateComponent(log.component);
                });
                actions.appendChild(btn);
            }

            if (log.node) {
                const btn = document.createElement('button');
                btn.className = 'log-action-btn';
                btn.textContent = `📍 Locate Node ${log.node}`;
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    locateNode(log.node);
                });
                actions.appendChild(btn);
            }

            if (log.line_number) {
                const btn = document.createElement('button');
                btn.className = 'log-action-btn';
                btn.textContent = `📄 Show Line ${log.line_number}`;
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    locateNetlistLine(log.line_number);
                });
                actions.appendChild(btn);
            }

            if (actions.children.length > 0) {
                row.appendChild(actions);
            }

            consoleBody.appendChild(row);
        });
    }

    function pushLogs(logs) {
        consoleLogs = logs || [];
        setConsoleFilter('all');
        
        const hasIssue = consoleLogs.some(log => log.type === 'error' || log.type === 'warning');
        if (hasIssue) {
            toggleConsole(true);
        }
    }

    // ═══════════════════════════════════════════
    // TEST PRESET CIRCUIT LOADER
    // ═══════════════════════════════════════════
    if (window.location.search.includes('test=1')) {
        components = [
            { id: "V1", type: "source", name: "V1", x: 100, y: 150, rotation: 0, params: { dc: "5" } },
            { id: "R1", type: "resistor", name: "R1", x: 200, y: 140, rotation: 90, params: { value: "1k" } },
            { id: "GND1", type: "ground", name: "GND1", x: 100, y: 240, rotation: 0, params: {} },
            { id: "GND2", type: "ground", name: "GND2", x: 200, y: 240, rotation: 0, params: {} }
        ];
        wires = [
            [{ x: 100, y: 110 }, { x: 200, y: 100 }],
            [{ x: 100, y: 190 }, { x: 100, y: 220 }],
            [{ x: 200, y: 180 }, { x: 200, y: 220 }]
        ];
        setTimeout(() => {
            if (typeof rerouteAllWires === 'function') {
                rerouteAllWires();
            }
            render();
        }, 100);
    }

    // ═══════════════════════════════════════════
    // INITIAL DRAW
    // ═══════════════════════════════════════════
    render();
});
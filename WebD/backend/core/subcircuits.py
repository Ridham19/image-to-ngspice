"""
core/subcircuits.py — Unified SPICE Subcircuit Library & Preset Catalog

Contains built-in SPICE .subckt definitions for:
  - Timers: NE555 Timer IC (astable/monostable)
  - Voltage Regulators: LM7805 (+5V), LM7812 (+12V), LM317 (Adj)
  - Analog Blocks: LM741 Op-Amp, Integrator, Differentiator,
    Summing Amplifier, Schmitt Trigger, Instrumentation Amplifier
  - Sequential Logic: SR Flip-Flop, JK Flip-Flop, D Flip-Flop, T Flip-Flop
  - Combinational Logic: Half Adder, Full Adder, 2:1 Multiplexer
  - Logic Gates: AND2, OR2, NOT1, NAND2, NOR2, XOR2, XNOR2

All subcircuits are modeled for seamless NGSPICE convergence using continuous
behavioral voltage/current sources (tanh functions) and standard analog macro-models.
"""

from typing import Dict, Any, List, Optional


# ═══════════════════════════════════════════════════════════════════════
# PRESET CATALOG METADATA (for UI dropdowns and auto-configuration)
# ═══════════════════════════════════════════════════════════════════════

SUBCIRCUIT_CATALOG: Dict[str, Dict[str, Any]] = {
    # ── Timers ──
    'NE555': {
        'name': 'NE555',
        'title': 'NE555 Precision Timer IC',
        'category': 'Timers & Oscillators',
        'num_pins': 8,
        'pins': ['GND', 'TRIG', 'OUT', 'RESET', 'CTRL', 'THRESH', 'DISCH', 'VCC'],
        'pin_descriptions': [
            'Pin 1: GND (0V reference)',
            'Pin 2: TRIG (Trigger, active low < 1/3 VCC)',
            'Pin 3: OUT (Output pin)',
            'Pin 4: RESET (Active-low reset, tie to VCC if unused)',
            'Pin 5: CTRL (Control voltage, nominal 2/3 VCC)',
            'Pin 6: THRESH (Threshold, active high > 2/3 VCC)',
            'Pin 7: DISCH (Open-collector discharge pin)',
            'Pin 8: VCC (Positive supply: +4.5V to +15V)',
        ],
        'default_component_type': 'ic',
    },

    # ── Voltage Regulators ──
    'LM7805': {
        'name': 'LM7805',
        'title': 'LM7805 +5V Linear Voltage Regulator',
        'category': 'Voltage Regulators',
        'num_pins': 3,
        'pins': ['IN', 'GND', 'OUT'],
        'pin_descriptions': [
            'Pin 1: IN (Unregulated DC input: 7V to 25V)',
            'Pin 2: GND (Ground reference 0V)',
            'Pin 3: OUT (Regulated +5.0V output)',
        ],
        'default_component_type': 'ic',
    },
    'LM7812': {
        'name': 'LM7812',
        'title': 'LM7812 +12V Linear Voltage Regulator',
        'category': 'Voltage Regulators',
        'num_pins': 3,
        'pins': ['IN', 'GND', 'OUT'],
        'pin_descriptions': [
            'Pin 1: IN (Unregulated DC input: 14.5V to 30V)',
            'Pin 2: GND (Ground reference 0V)',
            'Pin 3: OUT (Regulated +12.0V output)',
        ],
        'default_component_type': 'ic',
    },
    'LM317': {
        'name': 'LM317',
        'title': 'LM317 Adjustable Positive Linear Regulator',
        'category': 'Voltage Regulators',
        'num_pins': 3,
        'pins': ['IN', 'ADJ', 'OUT'],
        'pin_descriptions': [
            'Pin 1: IN (Input DC supply)',
            'Pin 2: ADJ (Voltage adjust pin, V_ref = 1.25V between OUT and ADJ)',
            'Pin 3: OUT (Regulated output: Vout = 1.25V * (1 + R2/R1))',
        ],
        'default_component_type': 'ic',
    },

    # ── Analog Operational Building Blocks ──
    'LM741': {
        'name': 'LM741',
        'title': 'LM741 General Purpose Operational Amplifier',
        'category': 'Amplifiers & Conditioners',
        'num_pins': 5,
        'pins': ['VPLUS', 'VMINUS', 'OUT', 'VSPOS', 'VSNEG'],
        'pin_descriptions': [
            'Pin 1: VPLUS (Non-inverting input +)',
            'Pin 2: VMINUS (Inverting input -)',
            'Pin 3: OUT (Amplifier output)',
            'Pin 4: VSPOS (Positive power rail +15V)',
            'Pin 5: VSNEG (Negative power rail -15V)',
        ],
        'default_component_type': 'opamp',
    },
    'INTEGRATOR': {
        'name': 'INTEGRATOR',
        'title': 'Active Miller Inverting Integrator',
        'category': 'Amplifiers & Conditioners',
        'num_pins': 4,
        'pins': ['IN', 'OUT', 'VCC', 'VEE'],
        'pin_descriptions': [
            'Pin 1: IN (Signal input)',
            'Pin 2: OUT (Integrated output: Vout = -1/(R*C) * integral(Vin))',
            'Pin 3: VCC (Positive power supply rail +15V)',
            'Pin 4: VEE (Negative power supply rail -15V)',
        ],
        'default_component_type': 'ic',
    },
    'DIFFERENTIATOR': {
        'name': 'DIFFERENTIATOR',
        'title': 'Practical Active Differentiator',
        'category': 'Amplifiers & Conditioners',
        'num_pins': 4,
        'pins': ['IN', 'OUT', 'VCC', 'VEE'],
        'pin_descriptions': [
            'Pin 1: IN (Signal input)',
            'Pin 2: OUT (Differentiated output: Vout = -R*C * dVin/dt)',
            'Pin 3: VCC (Positive power supply rail +15V)',
            'Pin 4: VEE (Negative power supply rail -15V)',
        ],
        'default_component_type': 'ic',
    },
    'SUMMING_AMP': {
        'name': 'SUMMING_AMP',
        'title': '3-Input Inverting Summing Amplifier',
        'category': 'Amplifiers & Conditioners',
        'num_pins': 6,
        'pins': ['IN1', 'IN2', 'IN3', 'OUT', 'VCC', 'VEE'],
        'pin_descriptions': [
            'Pin 1: IN1 (Input channel 1)',
            'Pin 2: IN2 (Input channel 2)',
            'Pin 3: IN3 (Input channel 3)',
            'Pin 4: OUT (Summed output: -(IN1 + IN2 + IN3))',
            'Pin 5: VCC (Positive rail +15V)',
            'Pin 6: VEE (Negative rail -15V)',
        ],
        'default_component_type': 'ic',
    },
    'SCHMITT_TRIGGER': {
        'name': 'SCHMITT_TRIGGER',
        'title': 'Inverting Schmitt Trigger with Hysteresis',
        'category': 'Amplifiers & Conditioners',
        'num_pins': 4,
        'pins': ['IN', 'OUT', 'VCC', 'VEE'],
        'pin_descriptions': [
            'Pin 1: IN (Signal input)',
            'Pin 2: OUT (Clean bistable square-wave output)',
            'Pin 3: VCC (Positive rail +5V or +15V)',
            'Pin 4: VEE (Negative rail 0V or -15V)',
        ],
        'default_component_type': 'ic',
    },
    'INSTRUMENTATION_AMP': {
        'name': 'INSTRUMENTATION_AMP',
        'title': '3-OpAmp Instrumentation Amplifier (High CMRR)',
        'category': 'Amplifiers & Conditioners',
        'num_pins': 6,
        'pins': ['INP', 'INN', 'REF', 'OUT', 'VCC', 'VEE'],
        'pin_descriptions': [
            'Pin 1: INP (Non-inverting differential input)',
            'Pin 2: INN (Inverting differential input)',
            'Pin 3: REF (Output voltage reference pin, tie to 0V)',
            'Pin 4: OUT (Amplified differential output: Gain = 10)',
            'Pin 5: VCC (Positive rail +15V)',
            'Pin 6: VEE (Negative rail -15V)',
        ],
        'default_component_type': 'ic',
    },

    # ── Digital Sequential Logic (Flip-Flops & Latches) ──
    'SR_FF': {
        'name': 'SR_FF',
        'title': 'SR Flip-Flop (Set-Reset Latch)',
        'category': 'Sequential Logic',
        'num_pins': 4,
        'pins': ['S', 'R', 'Q', 'QBAR'],
        'pin_descriptions': [
            'Pin 1: S (Set input, 5V active)',
            'Pin 2: R (Reset input, 5V active)',
            'Pin 3: Q (True output)',
            'Pin 4: QBAR (Complementary inverted output)',
        ],
        'default_component_type': 'ic',
    },
    'JK_FF': {
        'name': 'JK_FF',
        'title': 'JK Flip-Flop with Clock',
        'category': 'Sequential Logic',
        'num_pins': 5,
        'pins': ['J', 'K', 'CLK', 'Q', 'QBAR'],
        'pin_descriptions': [
            'Pin 1: J (J input: J=1, K=0 sets Q)',
            'Pin 2: K (K input: J=0, K=1 resets Q)',
            'Pin 3: CLK (Clock pulse: 0 to 5V)',
            'Pin 4: Q (True output)',
            'Pin 5: QBAR (Complementary output)',
        ],
        'default_component_type': 'ic',
    },
    'D_FF': {
        'name': 'D_FF',
        'title': 'D Flip-Flop / Data Register',
        'category': 'Sequential Logic',
        'num_pins': 4,
        'pins': ['D', 'CLK', 'Q', 'QBAR'],
        'pin_descriptions': [
            'Pin 1: D (Data input: 0V or 5V)',
            'Pin 2: CLK (Clock trigger pulse)',
            'Pin 3: Q (Stored data output)',
            'Pin 4: QBAR (Inverted output)',
        ],
        'default_component_type': 'ic',
    },
    'T_FF': {
        'name': 'T_FF',
        'title': 'Toggle Flip-Flop',
        'category': 'Sequential Logic',
        'num_pins': 4,
        'pins': ['T', 'CLK', 'Q', 'QBAR'],
        'pin_descriptions': [
            'Pin 1: T (Toggle control: T=1 toggles on clock pulse)',
            'Pin 2: CLK (Clock trigger pulse)',
            'Pin 3: Q (Toggled output)',
            'Pin 4: QBAR (Inverted output)',
        ],
        'default_component_type': 'ic',
    },

    # ── Digital Combinational Building Blocks ──
    'HALF_ADDER': {
        'name': 'HALF_ADDER',
        'title': '1-Bit Half Adder',
        'category': 'Combinational Logic',
        'num_pins': 4,
        'pins': ['A', 'B', 'SUM', 'CARRY'],
        'pin_descriptions': [
            'Pin 1: A (Operand bit A)',
            'Pin 2: B (Operand bit B)',
            'Pin 3: SUM (Sum output = A XOR B)',
            'Pin 4: CARRY (Carry output = A AND B)',
        ],
        'default_component_type': 'ic',
    },
    'FULL_ADDER': {
        'name': 'FULL_ADDER',
        'title': '1-Bit Full Adder with Carry-In',
        'category': 'Combinational Logic',
        'num_pins': 5,
        'pins': ['A', 'B', 'CIN', 'SUM', 'COUT'],
        'pin_descriptions': [
            'Pin 1: A (Operand bit A)',
            'Pin 2: B (Operand bit B)',
            'Pin 3: CIN (Carry-in input)',
            'Pin 4: SUM (Full sum output)',
            'Pin 5: COUT (Carry-out output)',
        ],
        'default_component_type': 'ic',
    },
    'MUX2TO1': {
        'name': 'MUX2TO1',
        'title': '2-to-1 Multiplexer',
        'category': 'Combinational Logic',
        'num_pins': 4,
        'pins': ['IN0', 'IN1', 'SEL', 'OUT'],
        'pin_descriptions': [
            'Pin 1: IN0 (Data input 0, selected when SEL < 2.5V)',
            'Pin 2: IN1 (Data input 1, selected when SEL >= 2.5V)',
            'Pin 3: SEL (Select control line)',
            'Pin 4: OUT (Multiplexed output)',
        ],
        'default_component_type': 'ic',
    },
}


# ═══════════════════════════════════════════════════════════════════════
# SPICE SUBCIRCUIT DEFINITIONS (.subckt ... .ends)
# ═══════════════════════════════════════════════════════════════════════

SUBCKT_SPICE_CODE: Dict[str, str] = {

    # ─────────────────────────────────────────────────────────────────
    # 1. NE555 Precision Timer IC
    # 8 Pins: GND(1) TRIG(2) OUT(3) RESET(4) CTRL(5) THRESH(6) DISCH(7) VCC(8)
    # ─────────────────────────────────────────────────────────────────
    'NE555': """\
* --- NE555 Precision Timer IC Subcircuit ---
* Pinout: 1=GND  2=TRIG  3=OUT  4=RESET  5=CTRL  6=THRESH  7=DISCH  8=VCC
.subckt NE555 gnd trig out reset ctrl thresh disch vcc

* Internal 3 x 5k Ohm voltage divider ladder
Rdiv1 vcc ctrl 5k
Rdiv2 ctrl v_lower 5k
Rdiv3 v_lower gnd 5k

* High impedance input buffers for threshold and trigger
Rin_thresh thresh gnd 100MEG
Rin_trig   trig   gnd 100MEG
Rin_reset  reset  gnd 100MEG

* Upper comparator: triggers when V(thresh) > V(ctrl)
* Lower comparator: triggers when V(trig) < V(v_lower)
B_comp_hi hi_raw 0 V = (1 + tanh(10 * (v(thresh) - v(ctrl)))) / 2
B_comp_lo lo_raw 0 V = (1 - tanh(10 * (v(trig) - v(v_lower)))) / 2
B_rst     rst_ok 0 V = (1 + tanh(10 * (v(reset) - 0.7))) / 2

* RS Latch memory core with continuous dynamic integration
* Int_q charges toward 5V when lo_raw is active, discharges when hi_raw or reset is active
B_latch_drv dq_dt 0 V = 5.0 * v(lo_raw) * v(rst_ok) - 5.0 * (v(hi_raw) + (1 - v(rst_ok))) * v(int_q) / 5.0
R_latch dq_dt latch_node 1k
C_latch latch_node 0 100p
B_q int_q 0 V = min(5.0, max(0.0, v(latch_node)))

* Push-Pull output stage (Pin 3: OUT)
B_out out_int 0 V = (v(int_q) / 5.0) * (v(vcc) - v(gnd)) * v(rst_ok)
R_out out_int out 10
C_out out gnd 10p

* Open-Collector Discharge Stage (Pin 7: DISCH)
* When Q is LOW (output low), discharge transistor conducts to GND
B_disch_trans disch gnd I = max(0.0, (1.0 - (v(int_q) / 5.0)) * (v(disch) - v(gnd)) / 2.0)

.ends NE555""",


    # ─────────────────────────────────────────────────────────────────
    # 2. LM7805 +5V Linear Voltage Regulator
    # 3 Pins: IN(1) GND(2) OUT(3)
    # ─────────────────────────────────────────────────────────────────
    'LM7805': """\
* --- LM7805 +5V Fixed Linear Voltage Regulator ---
* Pinout: 1=IN  2=GND  3=OUT
.subckt LM7805 in gnd out
Rin in gnd 1MEG
* Dropout voltage detection: V(in) - V(gnd) must be >= 7.0V for full 5V regulation
B_reg out_ideal gnd V = min(5.0, max(0.0, (v(in) - v(gnd) - 2.0)))
* Pass transistor output impedance and series resistance
R_pass out_ideal out 0.05
C_out out gnd 100n
R_bleed out gnd 10k
.ends LM7805""",


    # ─────────────────────────────────────────────────────────────────
    # 3. LM7812 +12V Linear Voltage Regulator
    # 3 Pins: IN(1) GND(2) OUT(3)
    # ─────────────────────────────────────────────────────────────────
    'LM7812': """\
* --- LM7812 +12V Fixed Linear Voltage Regulator ---
* Pinout: 1=IN  2=GND  3=OUT
.subckt LM7812 in gnd out
Rin in gnd 1MEG
* Dropout detection: input needs to be >= 14.0V for clean +12V regulation
B_reg out_ideal gnd V = min(12.0, max(0.0, (v(in) - v(gnd) - 2.0)))
R_pass out_ideal out 0.05
C_out out gnd 100n
R_bleed out gnd 20k
.ends LM7812""",


    # ─────────────────────────────────────────────────────────────────
    # 4. LM317 Adjustable Positive Linear Regulator
    # 3 Pins: IN(1) ADJ(2) OUT(3)
    # ─────────────────────────────────────────────────────────────────
    'LM317': """\
* --- LM317 Adjustable Positive Linear Regulator ---
* Pinout: 1=IN  2=ADJ  3=OUT
* Maintains 1.25V between OUT and ADJ: Vout = 1.25 * (1 + R2/R1) + Iadj*R2
.subckt LM317 in adj out
Rin in adj 1MEG
* Bandgap reference 1.25V above ADJ terminal, clamped by input headroom (Vin - 1.5V)
B_vref out_ideal 0 V = min(v(in) - 1.5, v(adj) + 1.25)
R_pass out_ideal out 0.08
I_adj adj 0 50u
C_out out 0 100n
.ends LM317""",


    # ─────────────────────────────────────────────────────────────────
    # 5. LM741 General Purpose Operational Amplifier
    # 5 Pins: vplus(1) vminus(2) out(3) vspos(4) vsneg(5)
    # ─────────────────────────────────────────────────────────────────
    'LM741': """\
* --- LM741 Operational Amplifier Subcircuit ---
* Pinout: 1=V+  2=V-  3=OUT  4=VS+  5=VS-
.subckt LM741 vplus vminus out vspos vsneg
Rin vplus vminus 2MEG
Rin_p vplus 0 100MEG
Rin_m vminus 0 100MEG
* High-gain linear differential stage with smooth supply rail clamping
B_diff int_node 0 V = max(v(vsneg) + 1.2, min(v(vspos) - 1.2, 200000 * (v(vplus) - v(vminus))))
R_out int_node out 75
C_comp out 0 20p
.ends LM741""",


    # ─────────────────────────────────────────────────────────────────
    # 6. Active Inverting Miller Integrator
    # 4 Pins: IN(1) OUT(2) VCC(3) VEE(4)
    # ─────────────────────────────────────────────────────────────────
    'INTEGRATOR': """\
* --- Active Inverting Integrator Subcircuit ---
* Pinout: 1=IN  2=OUT  3=VCC  4=VEE
* Transfer: Vout(t) = -1/(R*C) * integral(Vin dt), with R=10k, C=1uF (tau = 10ms)
* High value 10MEG parallel feedback resistor provides stable DC bias
.subckt INTEGRATOR in out vcc vee
R_in in sum_node 10k
C_fb sum_node out 1u
R_bleed sum_node out 10MEG
* Ideal virtual ground op-amp driver clamped to rails
B_op out_raw 0 V = max(v(vee) + 1.0, min(v(vcc) - 1.0, -100000 * v(sum_node)))
R_out out_raw out 20
C_out out 0 10p
.ends INTEGRATOR""",


    # ─────────────────────────────────────────────────────────────────
    # 7. Practical Active Differentiator
    # 4 Pins: IN(1) OUT(2) VCC(3) VEE(4)
    # ─────────────────────────────────────────────────────────────────
    'DIFFERENTIATOR': """\
* --- Practical Active Differentiator Subcircuit ---
* Pinout: 1=IN  2=OUT  3=VCC  4=VEE
* Transfer: Vout = -R_fb * C_in * d(Vin)/dt, with C_in=100nF, R_fb=10k
* Series R_stop (100 Ohm) prevents high-frequency noise peaking and oscillation
.subckt DIFFERENTIATOR in out vcc vee
R_stop in in_c 100
C_in in_c sum_node 100n
R_fb sum_node out 10k
C_fb sum_node out 100p
* Virtual ground gain block
B_op out_raw 0 V = max(v(vee) + 1.0, min(v(vcc) - 1.0, -100000 * v(sum_node)))
R_out out_raw out 20
C_out out 0 10p
.ends DIFFERENTIATOR""",


    # ─────────────────────────────────────────────────────────────────
    # 8. 3-Input Inverting Summing Amplifier
    # 6 Pins: IN1(1) IN2(2) IN3(3) OUT(4) VCC(5) VEE(6)
    # ─────────────────────────────────────────────────────────────────
    'SUMMING_AMP': """\
* --- 3-Input Inverting Summing Amplifier ---
* Pinout: 1=IN1  2=IN2  3=IN3  4=OUT  5=VCC  6=VEE
* Transfer: Vout = -(IN1 + IN2 + IN3) with unity weighted 10k resistors
.subckt SUMMING_AMP in1 in2 in3 out vcc vee
R1 in1 sum 10k
R2 in2 sum 10k
R3 in3 sum 10k
Rfb sum out 10k
B_amp out_raw 0 V = max(v(vee) + 1.0, min(v(vcc) - 1.0, -100000 * v(sum)))
R_out out_raw out 20
C_out out 0 10p
.ends SUMMING_AMP""",


    # ─────────────────────────────────────────────────────────────────
    # 9. Inverting Schmitt Trigger with Hysteresis
    # 4 Pins: IN(1) OUT(2) VCC(3) VEE(4)
    # ─────────────────────────────────────────────────────────────────
    'SCHMITT_TRIGGER': """\
* --- Inverting Schmitt Trigger Subcircuit ---
* Pinout: 1=IN  2=OUT  3=VCC  4=VEE
* Hysteresis thresholds set by internal feedback divider (V_HYS = +/- 0.5V)
.subckt SCHMITT_TRIGGER in out vcc vee
Rin in 0 10MEG
* Threshold reference based on output state
B_vth v_th 0 V = (v(vcc) + v(vee))/2.0 + 0.15 * (v(out) - (v(vcc) + v(vee))/2.0)
* High gain bistable transition
B_comp out_raw 0 V = ((v(vcc) + v(vee))/2.0) - ((v(vcc) - v(vee))/2.0) * tanh(20 * (v(in) - v(v_th)))
R_out out_raw out 50
C_out out 0 10p
.ends SCHMITT_TRIGGER""",


    # ─────────────────────────────────────────────────────────────────
    # 10. 3-OpAmp Instrumentation Amplifier
    # 6 Pins: INP(1) INN(2) REF(3) OUT(4) VCC(5) VEE(6)
    # ─────────────────────────────────────────────────────────────────
    'INSTRUMENTATION_AMP': """\
* --- 3-OpAmp Instrumentation Amplifier Subcircuit ---
* Pinout: 1=INP  2=INN  3=REF  4=OUT  5=VCC  6=VEE
* Fixed Gain = 10, High Input Impedance, High CMRR
.subckt INSTRUMENTATION_AMP inp inn ref out vcc vee
Rin_p inp 0 100MEG
Rin_n inn 0 100MEG
R_ref ref 0 100MEG
* Differential amplification with output reference addition
B_ina out_raw 0 V = max(v(vee) + 1.2, min(v(vcc) - 1.2, v(ref) + 10.0 * (v(inp) - v(inn))))
R_out out_raw out 25
C_out out 0 10p
.ends INSTRUMENTATION_AMP""",


    # ─────────────────────────────────────────────────────────────────
    # 11. SR Flip-Flop (Bistable Latch)
    # 4 Pins: S(1) R(2) Q(3) QBAR(4)
    # ─────────────────────────────────────────────────────────────────
    'SR_FF': """\
* --- SR Flip-Flop Subcircuit ---
* Pinout: 1=S  2=R  3=Q  4=QBAR
.subckt SR_FF s r q qbar
Rin_s s 0 100MEG
Rin_r r 0 100MEG
* State equation with cross-coupled NOR dynamics
B_q   q_raw   0 V = 5.0 * ( (1 + tanh(6 * (v(s) - 2.5)))/2 + (1 - (1 + tanh(6 * (v(r) - 2.5)))/2) * (v(q)/5.0) )
B_qbar qbar_raw 0 V = 5.0 - v(q_raw)
R_q q_raw q 50
R_qbar qbar_raw qbar 50
C_q q 0 20p
C_qbar qbar 0 20p
.ends SR_FF""",


    # ─────────────────────────────────────────────────────────────────
    # 12. JK Flip-Flop with Clock
    # 5 Pins: J(1) K(2) CLK(3) Q(4) QBAR(5)
    # ─────────────────────────────────────────────────────────────────
    'JK_FF': """\
* --- JK Flip-Flop Subcircuit ---
* Pinout: 1=J  2=K  3=CLK  4=Q  5=QBAR
.subckt JK_FF j k clk q qbar
Rin_j j 0 100MEG
Rin_k k 0 100MEG
Rin_c clk 0 100MEG
* Clock gating: active when CLK > 2.5V
* Next state: Q_next = J * ~Q + ~K * Q
B_nxt next_q 0 V = 5.0 * ( ((1 + tanh(6 * (v(j)-2.5)))/2) * (1 - v(q)/5.0) + (1 - (1 + tanh(6 * (v(k)-2.5)))/2) * (v(q)/5.0) )
B_clk_en clk_hi 0 V = (1 + tanh(8 * (v(clk) - 2.5))) / 2
* Latch update on clock enable
B_driver q_raw 0 V = v(clk_hi) * v(next_q) + (1 - v(clk_hi)) * v(q)
R_q q_raw q 50
C_q q 0 30p
B_qbar qbar_raw 0 V = 5.0 - v(q)
R_qbar qbar_raw qbar 50
C_qbar qbar 0 30p
.ends JK_FF""",


    # ─────────────────────────────────────────────────────────────────
    # 13. D Flip-Flop (Data Latch)
    # 4 Pins: D(1) CLK(2) Q(3) QBAR(4)
    # ─────────────────────────────────────────────────────────────────
    'D_FF': """\
* --- D Flip-Flop Subcircuit ---
* Pinout: 1=D  2=CLK  3=Q  4=QBAR
.subckt D_FF d clk q qbar
Rin_d d 0 100MEG
Rin_c clk 0 100MEG
B_clk_en en 0 V = (1 + tanh(8 * (v(clk) - 2.5))) / 2
* Follows D when CLK is high, latches state when CLK is low
B_d_latched q_raw 0 V = v(en) * (5.0 * (1 + tanh(6 * (v(d) - 2.5)))/2) + (1 - v(en)) * v(q)
R_q q_raw q 50
C_q q 0 30p
B_qbar qbar_raw 0 V = 5.0 - v(q)
R_qbar qbar_raw qbar 50
C_qbar qbar 0 30p
.ends D_FF""",


    # ─────────────────────────────────────────────────────────────────
    # 14. Toggle (T) Flip-Flop
    # 4 Pins: T(1) CLK(2) Q(3) QBAR(4)
    # ─────────────────────────────────────────────────────────────────
    'T_FF': """\
* --- Toggle (T) Flip-Flop Subcircuit ---
* Pinout: 1=T  2=CLK  3=Q  4=QBAR
.subckt T_FF t clk q qbar
Rin_t t 0 100MEG
Rin_c clk 0 100MEG
B_clk_en en 0 V = (1 + tanh(8 * (v(clk) - 2.5))) / 2
B_t_act  act 0 V = (1 + tanh(6 * (v(t) - 2.5))) / 2
B_target nxt 0 V = v(act) * (5.0 - v(q)) + (1 - v(act)) * v(q)
B_driver q_raw 0 V = v(en) * v(nxt) + (1 - v(en)) * v(q)
R_q q_raw q 50
C_q q 0 30p
B_qbar qbar_raw 0 V = 5.0 - v(q)
R_qbar qbar_raw qbar 50
C_qbar qbar 0 30p
.ends T_FF""",


    # ─────────────────────────────────────────────────────────────────
    # 15. 1-Bit Half Adder
    # 4 Pins: A(1) B(2) SUM(3) CARRY(4)
    # ─────────────────────────────────────────────────────────────────
    'HALF_ADDER': """\
* --- 1-Bit Half Adder Subcircuit ---
* Pinout: 1=A  2=B  3=SUM  4=CARRY
.subckt HALF_ADDER a b sum carry
Rin_a a 0 100MEG
Rin_b b 0 100MEG
* A_logic and B_logic (0 to 1 normalized)
B_an a_norm 0 V = (1 + tanh(5 * (v(a) - 2.5))) / 2
B_bn b_norm 0 V = (1 + tanh(5 * (v(b) - 2.5))) / 2
* SUM = A XOR B; CARRY = A AND B
B_sum_drv   sum_raw   0 V = 5.0 * ( v(a_norm)*(1 - v(b_norm)) + v(b_norm)*(1 - v(a_norm)) )
B_carry_drv carry_raw 0 V = 5.0 * ( v(a_norm) * v(b_norm) )
R_s sum_raw sum 50
C_s sum 0 10p
R_c carry_raw carry 50
C_c carry 0 10p
.ends HALF_ADDER""",


    # ─────────────────────────────────────────────────────────────────
    # 16. 1-Bit Full Adder
    # 5 Pins: A(1) B(2) CIN(3) SUM(4) COUT(5)
    # ─────────────────────────────────────────────────────────────────
    'FULL_ADDER': """\
* --- 1-Bit Full Adder Subcircuit ---
* Pinout: 1=A  2=B  3=CIN  4=SUM  5=COUT
.subckt FULL_ADDER a b cin sum cout
Rin_a a 0 100MEG
Rin_b b 0 100MEG
Rin_c cin 0 100MEG
B_an a_norm 0 V = (1 + tanh(5 * (v(a) - 2.5))) / 2
B_bn b_norm 0 V = (1 + tanh(5 * (v(b) - 2.5))) / 2
B_cn c_norm 0 V = (1 + tanh(5 * (v(cin) - 2.5))) / 2
* Intermediate: AxorB = A ^ B
B_x1 x1 0 V = v(a_norm)*(1 - v(b_norm)) + v(b_norm)*(1 - v(a_norm))
* SUM = AxorB ^ CIN
B_sum_drv sum_raw 0 V = 5.0 * ( v(x1)*(1 - v(c_norm)) + v(c_norm)*(1 - v(x1)) )
* COUT = (A & B) | (CIN & AxorB)
B_cout_drv cout_raw 0 V = 5.0 * min(1.0, (v(a_norm)*v(b_norm) + v(c_norm)*v(x1)))
R_s sum_raw sum 50
C_s sum 0 10p
R_c cout_raw cout 50
C_c cout 0 10p
.ends FULL_ADDER""",


    # ─────────────────────────────────────────────────────────────────
    # 17. 2-to-1 Multiplexer (MUX)
    # 4 Pins: IN0(1) IN1(2) SEL(3) OUT(4)
    # ─────────────────────────────────────────────────────────────────
    'MUX2TO1': """\
* --- 2-to-1 Multiplexer Subcircuit ---
* Pinout: 1=IN0  2=IN1  3=SEL  4=OUT
.subckt MUX2TO1 in0 in1 sel out
Rin0 in0 0 100MEG
Rin1 in1 0 100MEG
Rsel sel 0 100MEG
B_sn sel_norm 0 V = (1 + tanh(6 * (v(sel) - 2.5))) / 2
B_out out_raw 0 V = (1 - v(sel_norm)) * v(in0) + v(sel_norm) * v(in1)
R_out out_raw out 50
C_out out 0 10p
.ends MUX2TO1""",


    # ─────────────────────────────────────────────────────────────────
    # 18. Logic Gates (Smooth tanh behavioral subcircuits)
    # ─────────────────────────────────────────────────────────────────
    'and_gate': """\
* --- 2-Input AND Gate Subcircuit ---
.subckt AND2 in1 in2 out
Rin1 in1 0 100MEG
Rin2 in2 0 100MEG
B_out out_int 0 V = (( (1 + tanh(4*(v(in1)-2.5)))/2 ) * ( (1 + tanh(4*(v(in2)-2.5)))/2 )) * 5.0
R_out out_int out 50
C_out out 0 10p
.ends AND2""",

    'or_gate': """\
* --- 2-Input OR Gate Subcircuit ---
.subckt OR2 in1 in2 out
Rin1 in1 0 100MEG
Rin2 in2 0 100MEG
B_out out_int 0 V = (1 - (1 - (1 + tanh(4*(v(in1)-2.5)))/2) * (1 - (1 + tanh(4*(v(in2)-2.5)))/2)) * 5.0
R_out out_int out 50
C_out out 0 10p
.ends OR2""",

    'not_gate': """\
* --- NOT Gate / Inverter Subcircuit ---
.subckt NOT1 in out
Rin in 0 100MEG
B_out out_int 0 V = (1 - (1 + tanh(4*(v(in)-2.5)))/2) * 5.0
R_out out_int out 50
C_out out 0 10p
.ends NOT1""",

    'nand_gate': """\
* --- 2-Input NAND Gate Subcircuit ---
.subckt NAND2 in1 in2 out
Rin1 in1 0 100MEG
Rin2 in2 0 100MEG
B_out out_int 0 V = (1 - ((1 + tanh(4*(v(in1)-2.5)))/2) * ((1 + tanh(4*(v(in2)-2.5)))/2)) * 5.0
R_out out_int out 50
C_out out 0 10p
.ends NAND2""",

    'nor_gate': """\
* --- 2-Input NOR Gate Subcircuit ---
.subckt NOR2 in1 in2 out
Rin1 in1 0 100MEG
Rin2 in2 0 100MEG
B_out out_int 0 V = ((1 - (1 + tanh(4*(v(in1)-2.5)))/2) * (1 - (1 + tanh(4*(v(in2)-2.5)))/2)) * 5.0
R_out out_int out 50
C_out out 0 10p
.ends NOR2""",

    'xor_gate': """\
* --- 2-Input XOR Gate Subcircuit ---
.subckt XOR2 in1 in2 out
Rin1 in1 0 100MEG
Rin2 in2 0 100MEG
B_out out_int 0 V = ( ((1 + tanh(4*(v(in1)-2.5)))/2)*(1 - (1 + tanh(4*(v(in2)-2.5)))/2) + ((1 + tanh(4*(v(in2)-2.5)))/2)*(1 - (1 + tanh(4*(v(in1)-2.5)))/2) ) * 5.0
R_out out_int out 50
C_out out 0 10p
.ends XOR2""",

    'xnor_gate': """\
* --- 2-Input XNOR Gate Subcircuit ---
.subckt XNOR2 in1 in2 out
Rin1 in1 0 100MEG
Rin2 in2 0 100MEG
B_out out_int 0 V = ( 1 - ( ((1 + tanh(4*(v(in1)-2.5)))/2)*(1 - (1 + tanh(4*(v(in2)-2.5)))/2) + ((1 + tanh(4*(v(in2)-2.5)))/2)*(1 - (1 + tanh(4*(v(in1)-2.5)))/2) ) ) * 5.0
R_out out_int out 50
C_out out 0 10p
.ends XNOR2""",
}

# Alias mapping for backwards compatibility with component type keys
SUBCKT_DEFINITIONS: Dict[str, Optional[str]] = {
    'opamp': SUBCKT_SPICE_CODE['LM741'],
    'ic': None,
    'transformer': None,
    'and_gate': SUBCKT_SPICE_CODE['and_gate'],
    'or_gate': SUBCKT_SPICE_CODE['or_gate'],
    'not_gate': SUBCKT_SPICE_CODE['not_gate'],
    'nand_gate': SUBCKT_SPICE_CODE['nand_gate'],
    'nor_gate': SUBCKT_SPICE_CODE['nor_gate'],
    'xor_gate': SUBCKT_SPICE_CODE['xor_gate'],
    'xnor_gate': SUBCKT_SPICE_CODE['xnor_gate'],
}


# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def get_subcircuit_spice(name: str) -> Optional[str]:
    """
    Look up subcircuit definition by preset name or type key.
    Case-insensitive lookup.
    """
    if not name:
        return None
    name_clean = name.strip()
    # Direct match
    if name_clean in SUBCKT_SPICE_CODE:
        return SUBCKT_SPICE_CODE[name_clean]
    # Case-insensitive match
    for k, code in SUBCKT_SPICE_CODE.items():
        if k.lower() == name_clean.lower():
            return code
    return None


def get_subcircuit_info(name: str) -> Optional[Dict[str, Any]]:
    """Retrieve catalog metadata for a given subcircuit name."""
    if not name:
        return None
    name_clean = name.strip()
    if name_clean in SUBCIRCUIT_CATALOG:
        return SUBCIRCUIT_CATALOG[name_clean]
    for k, item in SUBCIRCUIT_CATALOG.items():
        if k.lower() == name_clean.lower():
            return item
    return None


def list_subcircuits() -> List[Dict[str, Any]]:
    """
    Return list of all built-in subcircuit catalog entries with their
    metadata and bundled SPICE code.
    """
    catalog_list = []
    for key, item in SUBCIRCUIT_CATALOG.items():
        entry = dict(item)
        entry['spice_code'] = SUBCKT_SPICE_CODE.get(key, '')
        catalog_list.append(entry)
    return catalog_list

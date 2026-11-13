"""
ANGSI JR - Geoscience & Reservoir Engineering Brain

Purpose:
- Teach GeoCopilot fundamental seismic / 4D / RE reasoning
- Prevent over-interpretation
- Provide recommendation logic
- Generate context-aware suggested questions
"""

# ============================================================
# 1. DOMAIN KNOWLEDGE
# ============================================================

DOMAIN_KNOWLEDGE = {

    "seismic_fundamentals": """
Seismic data image contrasts in subsurface elastic properties.

A seismic reflection occurs when a seismic wave encounters
a contrast in acoustic impedance.

Acoustic impedance:

AI = density * P-wave velocity
AI = rho * Vp

Important seismic concepts include:
- seismic trace
- inline and crossline
- two-way travel time (TWT)
- depth
- amplitude
- phase
- polarity
- frequency
- wavelength
- vertical and lateral resolution
- seismic velocity
- acoustic impedance
- reflection coefficient
- wavelet
- horizon
- fault
- migration
- seismic inversion

Observed seismic amplitude is affected by geology, fluids,
wavelet, processing, illumination, tuning and acquisition.

Seismic amplitude or colour must not automatically be interpreted
as fluid type.
""",

    "4d_fundamentals": """
4D seismic is time-lapse seismic.

An earlier 3D seismic survey is normally called the baseline.
A later repeated survey is called the monitor.

The physical reasoning is:

Production / injection
        ->
pressure and saturation change
        ->
rock and fluid elastic properties change
        ->
Vp, Vs and density may change
        ->
seismic amplitude, impedance and/or travel time may change.

The objective is not merely to find differences between two
seismic cubes.

The objective is to determine whether reliable seismic changes
contain information about reservoir behaviour.
""",

    "4d_monitoring_workflow": """
A robust 4D workflow is:

1. Define reservoir-management objective
2. Assess 4D feasibility
3. Acquire baseline seismic
4. Production/injection occurs
5. Acquire monitor seismic
6. Perform repeatability and seismic QC
7. Register/match baseline and monitor
8. Calculate 4D attributes
9. Perform inversion where justified
10. Interpret within geological context
11. Test mechanisms using rock physics
12. Compare with wells and production/injection data
13. Compare with reservoir simulation
14. Update geological/dynamic model where justified
15. History match
16. Recommend next technical actions
17. Quantify uncertainty

Interpretation must follow adequate seismic QC.
""",

    "4d_qc": """
A baseline-monitor difference is NOT automatically a reservoir signal.

Possible non-reservoir causes include:
- acquisition geometry differences
- source/receiver positioning
- navigation differences
- processing differences
- phase differences
- frequency differences
- amplitude scaling
- registration errors
- residual time shift
- wavelet differences
- illumination
- multiples
- environmental effects
- seismic noise

Useful QC measurements include:
- NRMS
- trace correlation
- amplitude comparison
- spectral comparison
- time shift
- registration quality

Always ask whether the observed change is larger and more coherent
than expected seismic non-repeatability.
""",

    "rock_physics": """
Rock physics connects reservoir change to seismic change.

Important properties:
Vp = P-wave velocity
Vs = S-wave velocity
rho = density

Acoustic impedance:

AI = rho * Vp

Reservoir changes capable of changing elastic properties include:
- pore pressure
- effective stress
- water saturation
- gas saturation
- fluid substitution
- gas coming out of solution
- temperature
- reservoir compaction
- geomechanical deformation

Never automatically assume:
- amplitude increase = water
- amplitude decrease = depletion
- low impedance = gas
- high impedance = water
- red = gas
- blue = water

The seismic response depends on lithology, porosity, mineralogy,
pressure, stress, fluids, saturation, temperature, rock stiffness
and seismic resolution.
""",

    "reservoir_engineering": """
Reservoir-engineering concepts relevant to 4D include:
- reservoir pressure
- depletion
- pressure support
- oil saturation
- water saturation
- gas saturation
- fluid contacts
- aquifer movement
- gas-cap movement
- water injection
- gas injection
- water/gas sweep
- injector-producer communication
- reservoir connectivity
- vertical communication
- compartmentalisation
- fault transmissibility
- bypassed or unswept hydrocarbons
- history matching
- forecasting

Production and well data provide strong temporal information
at specific locations.

4D seismic can provide additional spatial information between wells.

Production data + well data + geology + 4D seismic + simulation
are stronger than interpreting any one dataset alone.
""",

    "geology_connectivity": """
Geological architecture controls fluid movement.

Important controls include:
- faults
- fault throw
- fault transmissibility
- sealing/transmissive faults
- compartments
- barriers
- baffles
- stratigraphy
- facies
- reservoir-quality variation
- lateral connectivity
- vertical connectivity
- preferential flow paths

Important questions:
- Does the anomaly follow the reservoir?
- Does it stop near a fault?
- Does it cross a mapped fault?
- Does it follow stratigraphy?
- Does it connect an injector and producer?
- Does it remain within one compartment?
- Is its movement consistent with expected connectivity?

Observed fluid movement that differs from simulation may indicate
problems in permeability, fault transmissibility, connectivity or
geological geometry.
""",

    "4d_attributes": """
Useful time-lapse measurements include:
- baseline amplitude
- monitor amplitude
- amplitude difference
- RMS amplitude
- dRMS
- NRMS
- trace correlation
- time shift
- impedance change
- velocity change
- inverted elastic-property changes

NRMS helps assess repeatability and change magnitude.

Correlation measures similarity.

Time shifts may contain information about velocity changes,
including reservoir and geomechanical effects.

Amplitude and impedance may respond to pressure and saturation,
but require appropriate rock-physics calibration.

No single seismic attribute should automatically determine fluid type.
""",

    "history_matching": """
4D seismic may constrain reservoir history matching.

General workflow:

reservoir model
    ->
simulate pressure and saturation
    ->
rock-physics transformation
    ->
predict seismic response
    ->
compare predicted and observed 4D
    ->
update geological/reservoir parameters
    ->
repeat

Parameters that may be investigated include:
- porosity
- permeability
- fault transmissibility
- connectivity
- geological geometry
- saturation distribution

A good seismic match does NOT guarantee a unique reservoir model.

Different parameter combinations may reproduce similar observations.
""",

    "uncertainty": """
4D seismic interpretation is often non-unique.

Always distinguish:

OBSERVATION
INTERPRETATION
RECOMMENDATION

Example:

Observation:
A coherent anomaly terminates close to a mapped fault.

Possible interpretation:
Restricted cross-fault communication.

Alternative explanations:
- poorer reservoir quality
- inaccurate fault position
- illumination problem
- poor repeatability
- processing artefact

Recommendation:
Check pressure communication, seismic QC and alternative
fault-transmissibility assumptions.

Confidence:
Medium.

Never convert a plausible explanation into a measured fact.
"""
}


# ============================================================
# 2. SCIENTIFIC REASONING WORKFLOW
# ============================================================

INTERPRETATION_RULES = """
For technical 4D questions follow this sequence:

1. OBSERVATION
State what the supplied data actually show.

2. SEISMIC QC
Evaluate repeatability, NRMS, correlation, registration,
time shift, phase, frequency, amplitude matching,
acquisition differences, processing differences and noise.

3. GEOLOGICAL CONTEXT
Relate the observation to reservoir horizons, faults,
stratigraphy, facies, compartments and wells.

4. ROCK PHYSICS
Consider pressure, saturation, fluid substitution,
effective stress, compaction and temperature.

5. RESERVOIR ENGINEERING
Consider depletion, pressure support, fluid sweep,
connectivity, compartmentalisation, fault transmissibility
and possible bypassed/unswept areas.

6. ALTERNATIVE EXPLANATIONS
Provide competing hypotheses when the evidence is non-unique.

7. MODEL COMPARISON
Compare observations with simulated pressure, saturation,
fluid movement or synthetic seismic when available.

8. RECOMMENDATION
Prefer the lowest-risk technical investigation capable of
distinguishing between the hypotheses.

9. CONFIDENCE
Report HIGH, MEDIUM or LOW confidence and explain what
missing information would improve confidence.
"""


# ============================================================
# 3. SCIENTIFIC GUARDRAILS
# ============================================================

SCIENTIFIC_GUARDRAILS = """
Never invent:
- seismic measurements
- wells
- faults
- pressures
- saturations
- production/injection history
- fluid type
- reserves/resources
- geological features

A baseline-monitor difference does not automatically mean
reservoir change.

A strong 4D anomaly does not automatically prove:
- depletion
- water movement
- gas movement
- hydrocarbons

A weak 4D response does not prove remaining hydrocarbons.

A weak response may result from poor seismic sensitivity.

Seismic colour does not identify fluid type.

An anomaly stopping near a fault does not by itself prove
that the fault is sealing.

Correlation does not necessarily prove hydraulic communication.

A good seismic history match does not prove that the reservoir
model is unique.

Always communicate uncertainty.
"""


# ============================================================
# 4. RECOMMENDATION PATTERNS
# ============================================================

RECOMMENDATION_PATTERNS = {

    "anomaly_stops_at_fault": {
        "possible_meanings": [
            "Restricted cross-fault communication",
            "Reservoir-quality deterioration",
            "Incorrect fault geometry",
            "Seismic illumination or processing effect"
        ],
        "recommended_checks": [
            "Review pressure communication across the fault",
            "Test alternative fault-transmissibility multipliers",
            "Review seismic repeatability near the fault",
            "Compare alternative geological realizations"
        ]
    },

    "anomaly_crosses_sealing_fault": {
        "possible_meanings": [
            "Fault may be more transmissible than assumed",
            "Mapped fault position may be incorrect",
            "Alternative fluid-flow pathway",
            "Seismic resolution or smearing"
        ],
        "recommended_checks": [
            "Review fault interpretation",
            "Test higher fault transmissibility",
            "Compare pressure between compartments",
            "Inspect seismic resolution and illumination"
        ]
    },

    "injector_to_producer_response": {
        "possible_meanings": [
            "Possible injector-producer communication",
            "Possible advancing injected-fluid front",
            "Preferential flow path"
        ],
        "recommended_checks": [
            "Compare injection and production history",
            "Check pressure response",
            "Compare simulated saturation maps",
            "Review tracer data if available"
        ]
    },

    "simulation_has_sweep_but_seismic_does_not": {
        "possible_meanings": [
            "Predicted change may be below seismic detectability",
            "Rock-physics model may be incorrect",
            "Simulation saturation distribution may be incorrect",
            "Seismic repeatability may be inadequate"
        ],
        "recommended_checks": [
            "Calculate expected seismic response",
            "Check seismic detectability",
            "Review petroelastic model",
            "Review saturation/connectivity assumptions"
        ]
    },

    "seismic_response_outside_simulated_sweep": {
        "possible_meanings": [
            "Missing flow path",
            "Incorrect permeability distribution",
            "Incorrect fault transmissibility",
            "Incorrect geological geometry",
            "Non-reservoir seismic effect"
        ],
        "recommended_checks": [
            "Verify 4D QC",
            "Test alternative connectivity models",
            "Review fault model",
            "Update reservoir history match if justified"
        ]
    },

    "possible_unswept_region": {
        "possible_meanings": [
            "Possible unswept or weakly swept reservoir",
            "Weak seismic sensitivity",
            "Compartmentalisation",
            "Poor reservoir quality"
        ],
        "recommended_checks": [
            "Compare simulated saturation",
            "Review nearby well information",
            "Validate rock-physics detectability",
            "Assess geological connectivity"
        ]
    },

    "high_difference_poor_repeatability": {
        "possible_meanings": [
            "Acquisition/processing difference",
            "Registration problem",
            "Seismic noise",
            "Reservoir response mixed with 4D noise"
        ],
        "recommended_checks": [
            "Review NRMS",
            "Review correlation",
            "Perform registration QC",
            "Review phase/amplitude matching",
            "Do not interpret reservoir behaviour until QC improves"
        ]
    }
}


# ============================================================
# 5. SUGGESTED QUESTIONS
# ============================================================

SUGGESTED_QUESTIONS = {

    "general": [
        "What are the main changes between the baseline and monitor seismic?",
        "Where is the biggest seismic change?",
        "What could the seismic changes mean?",
        "How reliable is the 4D seismic response?",
        "What should I investigate next?"
    ],

    "4d": [
        "Where is the strongest 4D anomaly?",
        "Where is the weakest 4D response?",
        "What could explain the observed 4D anomalies?",
        "Could the changes be related to pressure or saturation?",
        "Are the anomalies spatially coherent?",
        "Which areas show unexpected reservoir behaviour?"
    ],

    "qc": [
        "Is the baseline and monitor seismic sufficiently repeatable?",
        "Where is NRMS highest?",
        "Where is seismic correlation lowest?",
        "Could poor repeatability explain the observed differences?",
        "Are there possible registration or time-shift problems?",
        "Which areas have insufficient QC for reservoir interpretation?"
    ],

    "reservoir": [
        "Where are potential unswept or bypassed areas?",
        "Where might pressure depletion be strongest?",
        "Is there evidence consistent with fluid movement?",
        "Which areas may have poor reservoir connectivity?",
        "Is there evidence consistent with compartmentalisation?",
        "Where should the reservoir model be reviewed?"
    ],

    "geology": [
        "Do any 4D anomalies terminate near faults?",
        "Do any anomalies cross mapped faults?",
        "Could faults be controlling fluid movement?",
        "Which compartments behave differently?",
        "Does the 4D response follow the reservoir structure?"
    ],

    "wells": [
        "Which wells are closest to the strongest 4D changes?",
        "How does the seismic response relate to the wells?",
        "Is there possible injector-producer communication?",
        "Which wells should be investigated further?",
        "Do the well data support the seismic interpretation?"
    ],

    "opportunities": [
        "Where are potential unswept or bypassed areas?",
        "Which area has the strongest evidence for further investigation?",
        "What are the highest-ranked opportunities?",
        "What evidence supports this opportunity?",
        "What additional evidence is needed before considering an infill well?",
        "Which opportunity has the highest uncertainty?"
    ],

    "rock_physics": [
        "Could this anomaly be caused by pressure change?",
        "Could this anomaly be caused by water saturation change?",
        "Could gas saturation explain this response?",
        "Can pressure and saturation effects be distinguished?",
        "What rock-physics information is missing?"
    ],

    "uncertainty": [
        "What are the biggest uncertainties in this interpretation?",
        "Which conclusion is least reliable?",
        "What additional data would reduce uncertainty the most?"
    ]
}


# ============================================================
# 6. CONTEXT-AWARE QUESTION GENERATOR
# ============================================================

def get_suggested_questions(evidence_type=None, max_questions=6):

    questions = []

    if not evidence_type or str(evidence_type).lower() == "none":
        questions = SUGGESTED_QUESTIONS["general"]

    else:

        evidence = str(evidence_type).lower()

        questions.extend(SUGGESTED_QUESTIONS["general"][:2])

        if "4d" in evidence:
            questions.extend(SUGGESTED_QUESTIONS["4d"])

        if "seismic qc" in evidence or evidence == "qc":
            questions.extend(SUGGESTED_QUESTIONS["qc"])

        if "horizon" in evidence:
            questions.extend(SUGGESTED_QUESTIONS["geology"])

        if "well" in evidence or "log" in evidence:
            questions.extend(SUGGESTED_QUESTIONS["wells"])

        if "opportun" in evidence or "candidate" in evidence:
            questions.extend(SUGGESTED_QUESTIONS["opportunities"])

        if "uncertainty" in evidence:
            questions.extend(SUGGESTED_QUESTIONS["uncertainty"])

    # Remove duplicate questions while preserving order
    questions = list(dict.fromkeys(questions))

    return questions[:max_questions]


# ============================================================
# 7. BUILD FULL GEOSCIENCE KNOWLEDGE
# ============================================================

def get_geoscience_knowledge():

    sections = []

    for topic, text in DOMAIN_KNOWLEDGE.items():

        sections.append(
            f"""
========================================
KNOWLEDGE TOPIC: {topic.upper()}
========================================

{text.strip()}
"""
        )

    sections.append(INTERPRETATION_RULES)
    sections.append(SCIENTIFIC_GUARDRAILS)

    return "\n".join(sections)


# ============================================================
# 8. SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = f"""
You are ANGSI JR GeoCopilot.

You are a technical decision-support assistant for:

- geophysics
- 4D seismic
- reservoir geoscience
- rock physics
- reservoir engineering

Your objective is to interpret ANGSI dashboard evidence using
scientifically defensible reasoning.

{get_geoscience_knowledge()}

When technical literature from the PDF knowledge library is supplied,
use that literature to support general scientific principles.

Use dashboard evidence for field-specific observations.

Always distinguish:

1. DASHBOARD OBSERVATION
2. LITERATURE-SUPPORTED PRINCIPLE
3. INTERPRETATION
4. ALTERNATIVE EXPLANATIONS
5. RECOMMENDATION

For technical answers use:

Observation
Seismic QC
Geoscience interpretation
Rock-physics hypotheses
Reservoir-engineering implications
Alternative explanations
Recommendation
Confidence
Missing information
Sources

Use clear language understandable by a reservoir engineer who may
not be a seismic specialist.

Never fabricate missing evidence.
"""


# ============================================================
# 9. TEST
# ============================================================

if __name__ == "__main__":

    print("ANGSI Geoscience Brain loaded successfully.")

    print("\nKnowledge topics:")

    for topic in DOMAIN_KNOWLEDGE:
        print(" -", topic)

    print("\nExample 4D questions:")

    for question in get_suggested_questions("4D attributes"):
        print(" -", question)
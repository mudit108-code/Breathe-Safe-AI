"""
BreatheSafe AI — Environmental & Respiratory Health Risk Agent
LangGraph-powered stateful workflow for air quality + medtech triage
"""

import streamlit as st
import time
import json
import random
import math
from datetime import datetime, timedelta
from typing import TypedDict, Annotated, List, Optional
from enum import Enum

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="BreatheSafe AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

:root {
    --bg: #0a0f1e;
    --surface: #111827;
    --surface2: #1a2235;
    --accent-green: #00ffc6;
    --accent-blue: #4f8ef7;
    --accent-amber: #ffb830;
    --accent-red: #ff4d6d;
    --accent-purple: #a78bfa;
    --text: #e2e8f0;
    --muted: #64748b;
    --border: #1e293b;
}

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    color: var(--text);
}

.stApp {
    background: var(--bg);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Cards */
.bs-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}

.bs-card-accent {
    border-left: 3px solid var(--accent-green);
}

/* AQI Badge */
.aqi-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.aqi-good    { background: #0a2e1e; color: #00ffc6; border: 1px solid #00ffc6; }
.aqi-moderate{ background: #2e2a0a; color: #ffb830; border: 1px solid #ffb830; }
.aqi-usg    { background: #2e1a0a; color: #ff9500; border: 1px solid #ff9500; }
.aqi-unhealthy{ background: #2e0a0a; color: #ff4d6d; border: 1px solid #ff4d6d; }
.aqi-vunhealthy{ background: #1e0a2e; color: #a78bfa; border: 1px solid #a78bfa; }
.aqi-hazardous{ background: #1e0000; color: #ff0000; border: 1px solid #ff0000; }

/* Risk meter */
.risk-bar-wrap { background: #1e293b; border-radius: 8px; height: 10px; overflow: hidden; margin: 6px 0 2px; }
.risk-bar-fill { height: 100%; border-radius: 8px; transition: width 0.6s ease; }

/* Node status */
.node-step {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.55rem 0;
    font-size: 0.88rem;
    border-bottom: 1px solid var(--border);
}
.node-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}
.dot-done    { background: var(--accent-green); box-shadow: 0 0 8px var(--accent-green); }
.dot-active  { background: var(--accent-blue);  box-shadow: 0 0 8px var(--accent-blue); animation: pulse 1s infinite; }
.dot-pending { background: var(--muted); }

@keyframes pulse {
    0%,100% { opacity: 1; } 50% { opacity: 0.4; }
}

/* Metric tile */
.metric-tile {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
}
.metric-label {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 2px;
}

/* Section headers */
.sec-header {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    margin: 1.2rem 0 0.5rem;
}

/* Recommendations */
.rec-item {
    background: var(--surface);
    border-left: 3px solid var(--accent-blue);
    border-radius: 0 8px 8px 0;
    padding: 0.65rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.88rem;
}
.rec-critical { border-left-color: var(--accent-red); }
.rec-warning  { border-left-color: var(--accent-amber); }
.rec-ok       { border-left-color: var(--accent-green); }

/* Alert box */
.alert-box {
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    font-size: 0.9rem;
    margin-bottom: 0.75rem;
}
.alert-critical { background: #2e0a0a; border: 1px solid var(--accent-red); }
.alert-warning  { background: #2e2a0a; border: 1px solid var(--accent-amber); }
.alert-info     { background: #0a1a2e; border: 1px solid var(--accent-blue); }
.alert-success  { background: #0a2e1e; border: 1px solid var(--accent-green); }

/* Graph flow visual */
.graph-node {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 5px 12px;
    font-size: 0.75rem;
    margin: 3px;
}
.graph-node.active { border-color: var(--accent-green); color: var(--accent-green); }

/* Button styling */
div.stButton > button {
    background: linear-gradient(135deg, #00ffc6 0%, #4f8ef7 100%);
    color: #0a0f1e;
    border: none;
    border-radius: 8px;
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    padding: 0.6rem 1.8rem;
    font-size: 0.9rem;
    width: 100%;
    cursor: pointer;
    transition: opacity 0.2s;
}
div.stButton > button:hover { opacity: 0.88; }

/* Selectbox / Slider */
div[data-baseweb="select"] > div {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
}
div[data-testid="stSlider"] label { color: var(--text) !important; }

/* Title */
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00ffc6, #4f8ef7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
}
.hero-sub {
    color: var(--muted);
    font-size: 0.85rem;
    margin-top: 2px;
}

.divider { border: none; border-top: 1px solid var(--border); margin: 1rem 0; }

/* Spinner override */
div[data-testid="stSpinner"] { color: var(--accent-green) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LANGGRAPH STATE DEFINITION
# ─────────────────────────────────────────────

class RiskLevel(str, Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    CRITICAL = "Critical"

class AgentState(TypedDict):
    # Environmental inputs
    city: str
    aqi: int
    pm25: float
    pm10: float
    no2: float
    co: float
    o3: float
    humidity: float
    temperature: float
    wind_speed: float

    # Patient health inputs
    age: int
    has_asthma: bool
    has_copd: bool
    has_heart_disease: bool
    is_smoker: bool
    outdoor_hours: float
    symptoms: List[str]

    # Computed state (updated by graph nodes)
    aqi_category: str
    env_risk_score: float
    patient_risk_score: float
    combined_risk: float
    risk_level: str
    pollutant_alerts: List[str]
    recommendations: List[dict]
    medical_flags: List[str]
    emergency_flag: bool
    exposure_forecast: List[dict]
    graph_trace: List[str]
    final_report: dict


# ─────────────────────────────────────────────
# LANGGRAPH NODE FUNCTIONS
# ─────────────────────────────────────────────

def node_ingest_env_data(state: AgentState) -> AgentState:
    """Node 1 — Ingest & validate environmental sensor data"""
    trace = state.get("graph_trace", [])
    trace.append("ingest_env_data")
    
    aqi = state["aqi"]
    if aqi <= 50:
        category = "Good"
    elif aqi <= 100:
        category = "Moderate"
    elif aqi <= 150:
        category = "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        category = "Unhealthy"
    elif aqi <= 300:
        category = "Very Unhealthy"
    else:
        category = "Hazardous"
    
    return {**state, "aqi_category": category, "graph_trace": trace}


def node_compute_env_risk(state: AgentState) -> AgentState:
    """Node 2 — Score environmental pollution risk"""
    trace = state.get("graph_trace", [])
    trace.append("compute_env_risk")
    alerts = []

    # Weighted pollutant scoring
    aqi_score = min(state["aqi"] / 300, 1.0) * 40
    pm25_score = min(state["pm25"] / 75, 1.0) * 20
    pm10_score = min(state["pm10"] / 150, 1.0) * 10
    no2_score  = min(state["no2"] / 200, 1.0) * 10
    co_score   = min(state["co"] / 10, 1.0) * 10
    o3_score   = min(state["o3"] / 180, 1.0) * 10

    env_risk = aqi_score + pm25_score + pm10_score + no2_score + co_score + o3_score

    # Threshold alerts
    if state["pm25"] > 35.4:
        alerts.append(f"⚠️ PM2.5 at {state['pm25']:.1f} µg/m³ — exceeds WHO 24h limit (15 µg/m³)")
    if state["pm10"] > 154:
        alerts.append(f"⚠️ PM10 at {state['pm10']:.1f} µg/m³ — Unhealthy range")
    if state["no2"] > 100:
        alerts.append(f"⚠️ NO₂ at {state['no2']:.1f} µg/m³ — above safe threshold")
    if state["o3"] > 100:
        alerts.append(f"⚠️ Ozone at {state['o3']:.1f} µg/m³ — elevated ground-level ozone")
    if state["co"] > 9:
        alerts.append(f"⚠️ CO at {state['co']:.1f} ppm — approaching danger level")
    if state["humidity"] > 80:
        alerts.append("💧 High humidity — promotes mold spore dispersion")

    return {**state, "env_risk_score": round(env_risk, 2), "pollutant_alerts": alerts, "graph_trace": trace}


def node_assess_patient_profile(state: AgentState) -> AgentState:
    """Node 3 — Evaluate patient's medical vulnerability"""
    trace = state.get("graph_trace", [])
    trace.append("assess_patient_profile")
    flags = []

    base = 0.0
    if state["age"] < 5 or state["age"] > 65:
        base += 20
        flags.append("Age-related vulnerability (pediatric or elderly)")
    if state["has_asthma"]:
        base += 25
        flags.append("Asthma — high PM2.5 & O₃ sensitivity")
    if state["has_copd"]:
        base += 30
        flags.append("COPD — severe risk from PM & NO₂ exposure")
    if state["has_heart_disease"]:
        base += 20
        flags.append("Cardiac condition — CO & PM2.5 cardiovascular stress risk")
    if state["is_smoker"]:
        base += 10
        flags.append("Active smoker — compounded respiratory burden")

    # Symptom severity
    symptom_weights = {
        "Coughing": 8, "Wheezing": 12, "Shortness of breath": 18,
        "Chest tightness": 15, "Eye irritation": 5, "Headache": 6,
        "Fatigue": 7, "Nausea": 8
    }
    for sym in state.get("symptoms", []):
        base += symptom_weights.get(sym, 5)

    # Outdoor exposure multiplier
    exposure_mult = 1 + (state["outdoor_hours"] / 10) * 0.4
    patient_risk = min(base * exposure_mult, 100)

    return {**state, "patient_risk_score": round(patient_risk, 2), "medical_flags": flags, "graph_trace": trace}


def node_fuse_risk_scores(state: AgentState) -> AgentState:
    """Node 4 — Fuse env + patient risk into combined score"""
    trace = state.get("graph_trace", [])
    trace.append("fuse_risk_scores")

    env  = state["env_risk_score"]
    pat  = state["patient_risk_score"]

    # Geometric mean weighted fusion
    combined = math.sqrt(env * 0.55 ** 2 + pat * 0.45 ** 2) * (1 + 0.1 if state["outdoor_hours"] > 6 else 1)
    combined = min(combined, 100)

    if combined < 25:
        level = RiskLevel.LOW.value
    elif combined < 50:
        level = RiskLevel.MODERATE.value
    elif combined < 75:
        level = RiskLevel.HIGH.value
    else:
        level = RiskLevel.CRITICAL.value

    return {**state, "combined_risk": round(combined, 2), "risk_level": level, "graph_trace": trace}


def node_generate_recommendations(state: AgentState) -> AgentState:
    """Node 5a — Generate personalised health & environmental recommendations"""
    trace = state.get("graph_trace", [])
    trace.append("generate_recommendations")
    recs = []
    level = state["risk_level"]

    # Air quality based
    if state["aqi"] > 150:
        recs.append({"text": "Avoid all non-essential outdoor activity.", "priority": "critical"})
    elif state["aqi"] > 100:
        recs.append({"text": "Limit strenuous outdoor exercise. Prefer morning hours.", "priority": "warning"})
    else:
        recs.append({"text": "Air quality is acceptable. Light outdoor activity is fine.", "priority": "ok"})

    # Mask recommendation
    if state["pm25"] > 35.4 or state["aqi"] > 150:
        recs.append({"text": "Wear N95/FFP2 mask if outdoors — PM2.5 is elevated.", "priority": "critical"})
    elif state["pm25"] > 12:
        recs.append({"text": "Consider wearing a mask outdoors especially near traffic.", "priority": "warning"})

    # Condition-specific
    if state["has_asthma"]:
        recs.append({"text": "Carry rescue inhaler at all times. Pre-medicate before going out.", "priority": "critical" if level in ["High", "Critical"] else "warning"})
        if state["o3"] > 70:
            recs.append({"text": "Ozone levels may trigger asthma. Keep windows closed midday.", "priority": "warning"})

    if state["has_copd"]:
        recs.append({"text": "COPD patients: monitor SpO₂ hourly. Consult pulmonologist if SpO₂ < 92%.", "priority": "critical"})
        recs.append({"text": "Use prescribed bronchodilator before any outdoor exposure.", "priority": "warning"})

    if state["has_heart_disease"]:
        if state["co"] > 5:
            recs.append({"text": "Elevated CO detected — avoid vehicle-heavy roads. Risk of cardiac stress.", "priority": "critical"})
        recs.append({"text": "Monitor pulse and BP after outdoor exposure. Rest if chest discomfort occurs.", "priority": "warning"})

    # Ventilation
    if state["humidity"] > 75:
        recs.append({"text": "High humidity indoors can increase mold. Use dehumidifier or AC.", "priority": "warning"})
    
    if state["wind_speed"] < 5 and state["aqi"] > 80:
        recs.append({"text": "Low wind — pollutants not dispersing. Stay indoors with HEPA air purifier.", "priority": "warning"})

    # Children / elderly
    if state["age"] < 12 or state["age"] > 70:
        recs.append({"text": "Vulnerable age group — limit outdoor time to < 30 min when AQI > 100.", "priority": "warning"})

    # General
    recs.append({"text": "Stay hydrated. Water helps your lungs filter particulates.", "priority": "ok"})
    if level in ["High", "Critical"]:
        recs.append({"text": "Keep emergency contacts and nearest hospital info readily accessible.", "priority": "warning"})

    return {**state, "recommendations": recs, "graph_trace": trace}


def node_emergency_escalation(state: AgentState) -> AgentState:
    """Node 5b — Emergency escalation path for critical cases"""
    trace = state.get("graph_trace", [])
    trace.append("emergency_escalation")

    critical_symptoms = {"Shortness of breath", "Chest tightness", "Wheezing"}
    sym_set = set(state.get("symptoms", []))
    is_emergency = (
        state["combined_risk"] >= 75 and
        (bool(sym_set & critical_symptoms) or state["has_copd"] or state["has_heart_disease"])
    ) or state["aqi"] >= 300

    recs = state.get("recommendations", [])
    if is_emergency:
        recs.insert(0, {"text": "🚨 EMERGENCY: Seek immediate medical attention or call emergency services.", "priority": "critical"})
        recs.insert(1, {"text": "Move patient to clean indoor air immediately. Do not leave alone.", "priority": "critical"})

    return {**state, "emergency_flag": is_emergency, "recommendations": recs, "graph_trace": trace}


def node_exposure_forecast(state: AgentState) -> AgentState:
    """Node 6 — Simulate 24-hour exposure forecast"""
    trace = state.get("graph_trace", [])
    trace.append("exposure_forecast")

    base_aqi = state["aqi"]
    forecast = []
    now = datetime.now()

    # Diurnal pollution pattern (peaks morning/evening rush, low midday wind)
    diurnal = [0.95, 0.90, 0.85, 0.82, 0.88, 1.05, 1.15, 1.20,
                1.10, 0.95, 0.88, 0.82, 0.80, 0.83, 0.90, 0.98,
                1.05, 1.18, 1.25, 1.15, 1.05, 1.02, 0.98, 0.96]

    for i in range(24):
        hour_time = (now + timedelta(hours=i)).strftime("%H:00")
        aqi_val = int(base_aqi * diurnal[i] * random.uniform(0.92, 1.08))
        aqi_val = max(5, min(aqi_val, 500))
        if aqi_val <= 50: cat = "Good"
        elif aqi_val <= 100: cat = "Moderate"
        elif aqi_val <= 150: cat = "USG"
        elif aqi_val <= 200: cat = "Unhealthy"
        else: cat = "Very Unhealthy"
        forecast.append({"hour": hour_time, "aqi": aqi_val, "category": cat})

    return {**state, "exposure_forecast": forecast, "graph_trace": trace}


def node_compile_report(state: AgentState) -> AgentState:
    """Node 7 — Compile final structured health report"""
    trace = state.get("graph_trace", [])
    trace.append("compile_report")

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "city": state["city"],
        "aqi": state["aqi"],
        "aqi_category": state["aqi_category"],
        "env_risk": state["env_risk_score"],
        "patient_risk": state["patient_risk_score"],
        "combined_risk": state["combined_risk"],
        "risk_level": state["risk_level"],
        "emergency": state["emergency_flag"],
        "pollutant_alerts": state["pollutant_alerts"],
        "medical_flags": state["medical_flags"],
        "recommendations": state["recommendations"],
        "forecast": state["exposure_forecast"],
    }
    return {**state, "final_report": report, "graph_trace": trace}


# ─────────────────────────────────────────────
# ROUTING LOGIC (conditional branching)
# ─────────────────────────────────────────────

def route_after_fusion(state: AgentState) -> str:
    """Conditional edge: if risk is critical or high with symptoms → emergency path"""
    critical_symptoms = {"Shortness of breath", "Chest tightness", "Wheezing"}
    sym_set = set(state.get("symptoms", []))
    if state["combined_risk"] >= 60 or (state["combined_risk"] >= 40 and bool(sym_set & critical_symptoms)):
        return "emergency_escalation"
    return "generate_recommendations"


# ─────────────────────────────────────────────
# BUILD LANGGRAPH WORKFLOW
# ─────────────────────────────────────────────

@st.cache_resource
def build_graph():
    g = StateGraph(AgentState)

    g.add_node("ingest_env_data",       node_ingest_env_data)
    g.add_node("compute_env_risk",      node_compute_env_risk)
    g.add_node("assess_patient_profile",node_assess_patient_profile)
    g.add_node("fuse_risk_scores",      node_fuse_risk_scores)
    g.add_node("generate_recommendations", node_generate_recommendations)
    g.add_node("emergency_escalation",  node_emergency_escalation)
    g.add_node("exposure_forecast",     node_exposure_forecast)
    g.add_node("compile_report",        node_compile_report)

    g.set_entry_point("ingest_env_data")
    g.add_edge("ingest_env_data",        "compute_env_risk")
    g.add_edge("compute_env_risk",       "assess_patient_profile")
    g.add_edge("assess_patient_profile", "fuse_risk_scores")

    # Conditional branching after risk fusion
    g.add_conditional_edges(
        "fuse_risk_scores",
        route_after_fusion,
        {
            "emergency_escalation":  "emergency_escalation",
            "generate_recommendations": "generate_recommendations",
        }
    )

    g.add_edge("emergency_escalation",  "exposure_forecast")
    g.add_edge("generate_recommendations", "exposure_forecast")
    g.add_edge("exposure_forecast",     "compile_report")
    g.add_edge("compile_report",        END)

    return g.compile()


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def aqi_class(aqi):
    if aqi <= 50:   return "aqi-good"
    if aqi <= 100:  return "aqi-moderate"
    if aqi <= 150:  return "aqi-usg"
    if aqi <= 200:  return "aqi-unhealthy"
    if aqi <= 300:  return "aqi-vunhealthy"
    return "aqi-hazardous"

def risk_color(level):
    return {"Low": "#00ffc6", "Moderate": "#ffb830", "High": "#ff9500", "Critical": "#ff4d6d"}.get(level, "#64748b")

def rec_class(priority):
    return {"critical": "rec-critical", "warning": "rec-warning", "ok": "rec-ok"}.get(priority, "")

NODE_LABELS = {
    "ingest_env_data":         "📡 Ingest Environmental Data",
    "compute_env_risk":        "🌫️ Compute Pollution Risk",
    "assess_patient_profile":  "🫁 Assess Patient Profile",
    "fuse_risk_scores":        "⚖️ Fuse Risk Scores",
    "generate_recommendations":"💊 Generate Recommendations",
    "emergency_escalation":    "🚨 Emergency Escalation",
    "exposure_forecast":       "📈 24h Exposure Forecast",
    "compile_report":          "📋 Compile Health Report",
}

ALL_NODES = list(NODE_LABELS.keys())

# ─────────────────────────────────────────────
# SIDEBAR — INPUTS
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="hero-title">🫁 BreatheSafe AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">EnviroMed Risk Intelligence</div>', unsafe_allow_html=True)
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    st.markdown('<div class="sec-header">📍 Location & Environment</div>', unsafe_allow_html=True)
    city = st.text_input("City", value="Hyderabad")
    col1, col2 = st.columns(2)
    with col1:
        aqi  = st.slider("AQI",  0, 500, 142)
        pm25 = st.slider("PM2.5 (µg/m³)", 0.0, 200.0, 42.5)
        pm10 = st.slider("PM10 (µg/m³)",  0.0, 300.0, 88.0)
        no2  = st.slider("NO₂ (µg/m³)",   0.0, 400.0, 85.0)
    with col2:
        co   = st.slider("CO (ppm)", 0.0, 20.0, 4.2)
        o3   = st.slider("O₃ (µg/m³)", 0.0, 300.0, 95.0)
        humidity    = st.slider("Humidity (%)", 0, 100, 72)
        wind_speed  = st.slider("Wind Speed (km/h)", 0.0, 60.0, 8.0)
    temperature = st.slider("Temperature (°C)", -10, 50, 34)

    st.markdown('<div class="sec-header">👤 Patient Profile</div>', unsafe_allow_html=True)
    age = st.slider("Age", 1, 100, 42)
    col3, col4 = st.columns(2)
    with col3:
        has_asthma       = st.checkbox("Asthma", value=True)
        has_copd         = st.checkbox("COPD")
    with col4:
        has_heart_disease= st.checkbox("Heart Disease")
        is_smoker        = st.checkbox("Smoker")

    outdoor_hours = st.slider("Daily Outdoor Hours", 0.0, 16.0, 3.5)
    symptoms = st.multiselect(
        "Current Symptoms",
        ["Coughing", "Wheezing", "Shortness of breath", "Chest tightness",
         "Eye irritation", "Headache", "Fatigue", "Nausea"],
        default=["Coughing", "Wheezing"]
    )

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🔍  Run AI Analysis")


# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────

st.markdown('<div class="hero-title">BreatheSafe AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Stateful LangGraph agent for environmental & respiratory health risk assessment</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Graph architecture overview
with st.expander("🗂️ LangGraph Workflow Architecture", expanded=False):
    st.markdown("""
    <div class="bs-card">
    <div class="sec-header">Graph Nodes & Conditional Routing</div>
    <p style="font-size:0.85rem; color:#64748b; margin-bottom:0.75rem;">
    The agent uses a <b>StateGraph</b> with conditional branching — after fusing environmental and patient risk scores,
    the graph dynamically routes to either a standard recommendation path or an emergency escalation path.
    </p>
    <div>
      <span class="graph-node active">📡 Ingest Env</span> →
      <span class="graph-node active">🌫️ Env Risk</span> →
      <span class="graph-node active">🫁 Patient Profile</span> →
      <span class="graph-node active">⚖️ Fuse Scores</span> →
      <span style="color:#ffb830;font-size:0.8rem;"> ⬇️ conditional </span><br>
      <span class="graph-node" style="margin-left:1.5rem;">💊 Recommendations</span>
      <span style="color:#64748b;font-size:0.8rem;"> OR </span>
      <span class="graph-node" style="border-color:#ff4d6d;color:#ff4d6d;">🚨 Emergency Escalation</span> →
      <span class="graph-node active">📈 Forecast</span> →
      <span class="graph-node active">📋 Report</span>
    </div>
    </div>
    """, unsafe_allow_html=True)


if run_btn:
    graph = build_graph()

    initial_state: AgentState = {
        "city": city, "aqi": aqi, "pm25": pm25, "pm10": pm10,
        "no2": no2, "co": co, "o3": o3,
        "humidity": humidity, "temperature": temperature, "wind_speed": wind_speed,
        "age": age, "has_asthma": has_asthma, "has_copd": has_copd,
        "has_heart_disease": has_heart_disease, "is_smoker": is_smoker,
        "outdoor_hours": outdoor_hours, "symptoms": symptoms,
        "aqi_category": "", "env_risk_score": 0.0, "patient_risk_score": 0.0,
        "combined_risk": 0.0, "risk_level": "", "pollutant_alerts": [],
        "recommendations": [], "medical_flags": [], "emergency_flag": False,
        "exposure_forecast": [], "graph_trace": [], "final_report": {},
    }

    # Progress display
    col_prog, col_main = st.columns([1, 2.5])

    with col_prog:
        st.markdown('<div class="sec-header">⚙️ Graph Execution</div>', unsafe_allow_html=True)
        node_placeholders = {}
        for nk in ALL_NODES:
            node_placeholders[nk] = st.empty()
            node_placeholders[nk].markdown(
                f'<div class="node-step"><div class="node-dot dot-pending"></div><span style="color:#64748b">{NODE_LABELS[nk]}</span></div>',
                unsafe_allow_html=True
            )

    with col_main:
        result_placeholder = st.empty()

    # Run graph with streaming node updates
    result = None
    for event in graph.stream(initial_state):
        for node_name, node_state in event.items():
            if node_name in node_placeholders:
                # Mark completed nodes
                for prev in node_state.get("graph_trace", [])[:-1]:
                    if prev in node_placeholders:
                        node_placeholders[prev].markdown(
                            f'<div class="node-step"><div class="node-dot dot-done"></div><span>{NODE_LABELS[prev]}</span></div>',
                            unsafe_allow_html=True
                        )
                # Mark active node
                node_placeholders[node_name].markdown(
                    f'<div class="node-step"><div class="node-dot dot-active"></div><b>{NODE_LABELS[node_name]}</b></div>',
                    unsafe_allow_html=True
                )
                time.sleep(0.25)
            result = node_state

    # Mark all done
    if result:
        for nk in result.get("graph_trace", []):
            if nk in node_placeholders:
                node_placeholders[nk].markdown(
                    f'<div class="node-step"><div class="node-dot dot-done"></div><span>{NODE_LABELS[nk]}</span></div>',
                    unsafe_allow_html=True
                )

    # ─────────────────────────────────────────────
    # RESULTS DISPLAY
    # ─────────────────────────────────────────────
    if result and result.get("final_report"):
        rep = result["final_report"]
        rl  = rep["risk_level"]
        rc  = risk_color(rl)

        with col_main:
            # Emergency banner
            if rep["emergency"]:
                st.markdown(f"""
                <div class="alert-box alert-critical">
                🚨 <b>EMERGENCY ALERT</b> — This patient profile requires <b>immediate medical attention</b>.
                High environmental exposure combined with critical health conditions detected.
                </div>""", unsafe_allow_html=True)

            # Key metrics row
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""<div class="metric-tile">
                <div class="metric-value" style="color:{rc}">{rep['combined_risk']:.0f}</div>
                <div class="metric-label">Combined Risk Score</div>
                <span class="aqi-badge" style="color:{rc};border:1px solid {rc};background:transparent;margin-top:4px;display:inline-block">{rl}</span>
                </div>""", unsafe_allow_html=True)
            with m2:
                ac = aqi_class(rep["aqi"])
                st.markdown(f"""<div class="metric-tile">
                <div class="metric-value" style="color:#4f8ef7">{rep['aqi']}</div>
                <div class="metric-label">Air Quality Index</div>
                <span class="aqi-badge {ac}" style="margin-top:4px;display:inline-block">{rep['aqi_category'][:12]}</span>
                </div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""<div class="metric-tile">
                <div class="metric-value" style="color:#00ffc6">{rep['env_risk']:.0f}</div>
                <div class="metric-label">Env Risk Score</div>
                <div class="risk-bar-wrap"><div class="risk-bar-fill" style="width:{rep['env_risk']}%;background:linear-gradient(90deg,#00ffc6,#4f8ef7)"></div></div>
                </div>""", unsafe_allow_html=True)
            with m4:
                st.markdown(f"""<div class="metric-tile">
                <div class="metric-value" style="color:#a78bfa">{rep['patient_risk']:.0f}</div>
                <div class="metric-label">Patient Risk Score</div>
                <div class="risk-bar-wrap"><div class="risk-bar-fill" style="width:{rep['patient_risk']}%;background:linear-gradient(90deg,#a78bfa,#ff4d6d)"></div></div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            tab1, tab2, tab3, tab4 = st.tabs(["💊 Recommendations", "🌫️ Pollutant Alerts", "🏥 Medical Flags", "📈 24h Forecast"])

            with tab1:
                for r in rep["recommendations"]:
                    st.markdown(f'<div class="rec-item {rec_class(r["priority"])}">{r["text"]}</div>', unsafe_allow_html=True)

            with tab2:
                if rep["pollutant_alerts"]:
                    for a in rep["pollutant_alerts"]:
                        st.markdown(f'<div class="alert-box alert-warning">{a}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="alert-box alert-success">✅ All pollutant levels within acceptable thresholds.</div>', unsafe_allow_html=True)

                # Pollutant mini-table
                st.markdown('<div class="sec-header">Input Readings</div>', unsafe_allow_html=True)
                pd_data = {
                    "Pollutant": ["PM2.5", "PM10", "NO₂", "CO", "O₃"],
                    "Value": [f"{pm25} µg/m³", f"{pm10} µg/m³", f"{no2} µg/m³", f"{co} ppm", f"{o3} µg/m³"],
                    "WHO Limit": ["15 µg/m³", "45 µg/m³", "25 µg/m³", "4 ppm", "100 µg/m³"],
                }
                import pandas as pd
                st.dataframe(pd.DataFrame(pd_data), hide_index=True, use_container_width=True)

            with tab3:
                if rep["medical_flags"]:
                    for f in rep["medical_flags"]:
                        st.markdown(f'<div class="alert-box alert-warning">⚕️ {f}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="alert-box alert-success">✅ No significant medical risk factors detected.</div>', unsafe_allow_html=True)

            with tab4:
                import pandas as pd
                fc_df = pd.DataFrame(rep["forecast"])
                st.markdown('<div class="sec-header">Projected AQI over next 24 hours</div>', unsafe_allow_html=True)
                st.line_chart(fc_df.set_index("hour")["aqi"], use_container_width=True, color="#00ffc6")
                with st.expander("View hourly table"):
                    st.dataframe(fc_df[["hour", "aqi", "category"]], hide_index=True, use_container_width=True)

            # Graph trace badge
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            branch = "🚨 Emergency Path" if rep["emergency"] else "💊 Standard Path"
            st.markdown(f"""
            <div class="bs-card bs-card-accent" style="padding:0.75rem 1rem">
            <span style="font-size:0.75rem;color:#64748b">LangGraph Branch Taken: </span>
            <b style="color:#00ffc6">{branch}</b>
            &nbsp;|&nbsp;
            <span style="font-size:0.75rem;color:#64748b">Nodes executed: </span>
            <b style="color:#4f8ef7">{len(rep.get('pollutant_alerts',[]))} alerts</b>
            &nbsp;|&nbsp;
            <span style="font-size:0.75rem;color:#64748b">Generated: </span>
            <span style="font-family:'Space Mono',monospace;font-size:0.75rem">{rep['generated_at']}</span>
            </div>""", unsafe_allow_html=True)

else:
    # Landing state
    st.markdown("""
    <div class="bs-card" style="text-align:center;padding:3rem 2rem">
    <div style="font-size:3rem;margin-bottom:1rem">🫁</div>
    <div style="font-size:1.1rem;font-weight:600;margin-bottom:0.5rem"><span style="color:#00ffc6">Run AI Analysis</span></div>
    <div style="color:#64748b;font-size:0.88rem;max-width:480px;margin:0 auto">
    BreatheSafe AI uses a stateful LangGraph workflow with conditional branching to assess real-time air quality data 
    against patient health profiles — generating personalised respiratory health recommendations and 24-hour exposure forecasts.
    </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""<div class="bs-card">
        <div style="font-size:1.4rem">🌫️</div>
        <div style="font-weight:600;margin:0.4rem 0 0.3rem">EnviroSensing</div>
        <div style="color:#64748b;font-size:0.82rem">Ingests AQI, PM2.5, PM10, NO₂, CO, O₃, humidity, and wind — computes weighted pollution risk score</div>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown("""<div class="bs-card">
        <div style="font-size:1.4rem">⚖️</div>
        <div style="font-weight:600;margin:0.4rem 0 0.3rem">Risk Fusion</div>
        <div style="color:#64748b;font-size:0.82rem">Geometric fusion of environmental & patient vulnerability scores with conditional graph routing</div>
        </div>""", unsafe_allow_html=True)
    with col_c:
        st.markdown("""<div class="bs-card">
        <div style="font-size:1.4rem">🚨</div>
        <div style="font-weight:600;margin:0.4rem 0 0.3rem">Smart Branching</div>
        <div style="color:#64748b;font-size:0.82rem">LangGraph conditional edges route to emergency escalation or standard recommendations based on combined risk</div>
        </div>""", unsafe_allow_html=True)

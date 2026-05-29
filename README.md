# 🫁 BreatheSafe AI
### Environmental & Respiratory Health Risk Intelligence — Powered by LangGraph


---

## 🌍 Real-World Use Case

**Problem:** Urban populations — especially those with respiratory or cardiovascular conditions — have no personalized, real-time tool that combines *environmental air quality data* with *individual health profiles* to produce actionable, condition-specific health guidance.

**Solution:** BreatheSafe AI is an **EnviroMed intelligence agent** that:
- Ingests multi-pollutant sensor data (AQI, PM2.5, PM10, NO₂, CO, O₃)
- Evaluates patient vulnerability (age, COPD, asthma, cardiac conditions, symptoms)
- Fuses both risk dimensions through a stateful LangGraph pipeline
- **Conditionally branches** into emergency escalation or standard recommendations
- Generates a 24-hour exposure forecast with diurnal pollution modelling

---



### Key LangGraph Features Used

| Feature | Implementation |
|---|---|
| `StateGraph` | Typed `AgentState` (TypedDict) flows through all nodes |
| Conditional Edges | `route_after_fusion()` branches to emergency or standard path |
| Node Functions | 8 pure-function nodes, each updating a slice of shared state |
| `graph.stream()` | Real-time node-by-node execution visible in UI |
| `@st.cache_resource` | Graph compiled once and reused across sessions |

---

## 🚀 Local Setup

```bash
git clone 
cd 

pip install -r requirements.txt

streamlit run app.py
```

**Python 3.10+ required**

---



## 🔬 Technical Highlights

- **No external APIs** — fully self-contained, runs offline
- **Typed state** — `AgentState` TypedDict ensures type-safe state mutations across nodes
- **Weighted risk fusion** — geometric mean model balances environmental and patient scores
- **Diurnal pollution modeling** — 24-hour AQI forecast uses real-world traffic/wind patterns
- **Condition-specific logic** — COPD, asthma, cardiac disease each trigger targeted guidance


---

## 🏥 Health Conditions Modelled

| Condition | Risk Modifier | Key Pollutant |
|---|---|---|
| Asthma | +25 pts | PM2.5, O₃ |
| COPD | +30 pts | PM, NO₂ |
| Heart Disease | +20 pts | CO, PM2.5 |
| Active Smoker | +10 pts | All |
| Age < 5 or > 65 | +20 pts | All |

---

## 📊 Pollutant Thresholds (WHO Guidelines)

| Pollutant | Safe Limit | Unit |
|---|---|---|
| PM2.5 | 15 | µg/m³ (24h) |
| PM10 | 45 | µg/m³ (24h) |
| NO₂ | 25 | µg/m³ (24h) |
| O₃ | 100 | µg/m³ (8h) |
| CO | 4 | ppm (24h) |

---



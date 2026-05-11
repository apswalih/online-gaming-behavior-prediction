# =============================================================================
# GAMING ENGAGEMENT LEVEL PREDICTION - STREAMLIT UI
# =============================================================================

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Gaming Engagement Level Prediction",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# MODEL LOADING
# =============================================================================
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "xgboost_pipeline.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# =============================================================================
# SESSION STATE
# =============================================================================
qp = st.query_params
if "about" in qp:
    st.session_state.show_about = (qp.get("about") == "1")
elif "show_about" not in st.session_state:
    st.session_state.show_about = False

about_href = "?about=0" if st.session_state.show_about else "?about=1"

# =============================================================================
# GLOBAL CSS
# =============================================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

header[data-testid="stHeader"],
div[data-testid="stToolbar"],
div[data-testid="stDecoration"],
div[data-testid="stStatusWidget"],
#MainMenu, footer {{ display: none !important; height: 0 !important; }}
.stApp > header {{ background: transparent !important; }}

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

.stApp {{
    background:
      radial-gradient(1200px 600px at 10% -10%, rgba(0,229,255,0.10), transparent 60%),
      radial-gradient(1000px 700px at 110% 10%, rgba(139,92,246,0.12), transparent 60%),
      radial-gradient(800px 600px at 50% 120%, rgba(34,197,94,0.08), transparent 60%),
      linear-gradient(180deg, #05060f 0%, #0a0b1e 100%);
    color: #e6e9f2;
}}

.stApp.about-open {{
    transform: translateX(-220px);
    filter: brightness(0.85);
}}

.block-container {{ max-width: 1400px; padding-top: 1rem; padding-bottom: 4rem; }}

.nav-row {{
    display: grid;
    grid-template-columns: minmax(0, 6fr) minmax(150px, 1fr);
    gap: 18px;
    align-items: stretch;
    margin-bottom: 20px;
}}

.topnav {{
    display:flex; align-items:center; justify-content:space-between;
    padding: 14px 22px; border-radius: 18px;
    background: linear-gradient(180deg, rgba(15,18,38,0.7), rgba(15,18,38,0.4));
    backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,0.06);
    height: 80px;
    min-height: 80px;
    box-sizing: border-box;
}}
.topnav-brand {{ display:flex; align-items:center; gap:12px; font-weight:800; letter-spacing:0.04em; }}
.topnav-brand .logo {{
    width:36px; height:36px; border-radius:12px;
    background: conic-gradient(from 180deg at 50% 50%, #00e5ff, #8b5cf6, #22c55e, #00e5ff);
    box-shadow: 0 8px 24px rgba(139,92,246,0.4);
}}
.topnav-brand .name {{
    font-family:'Space Grotesk', sans-serif;
    font-size: 1rem; color:#fff;
}}
.topnav-tag {{
    font-family:'JetBrains Mono', monospace;
    font-size:.72rem; letter-spacing:.18em; color:#94a3b8;
    text-transform: uppercase;
}}

.about-trigger {{
    height: 80px;
    min-height: 80px;
    border-radius: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-decoration: none !important;
    color: #e6e9f2 !important;
    font-family:'Space Grotesk', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    background: linear-gradient(180deg, rgba(15,18,38,0.7), rgba(15,18,38,0.4));
    backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow:
      inset 0 1px 0 rgba(255,255,255,0.05),
      0 14px 34px rgba(0,0,0,0.22);
    transition: border-color .18s ease, box-shadow .18s ease, background .18s ease;
}}
.about-trigger:hover {{
    color: #ffffff !important;
    background: linear-gradient(180deg, rgba(16,22,48,0.92), rgba(8,12,28,0.96));
    border-color: rgba(0,229,255,0.45);
    box-shadow:
      inset 0 1px 0 rgba(255,255,255,0.07),
      0 0 24px rgba(0,229,255,0.18);
}}

.about-overlay {{
    position: fixed;
    inset: 0;
    background: rgba(2,4,10,0.58);
    backdrop-filter: blur(5px);
    z-index: 9998;
}}
.about-overlay a.full {{
    position:absolute;
    inset:0;
    display:block;
}}

.about-panel {{
    position: fixed;
    top: 0;
    right: 0;
    height: 100vh;
    width: min(520px, 94vw);
    z-index: 9999;
    overflow-y: auto;
    padding: 28px;
    background:
      radial-gradient(700px 320px at 15% 0%, rgba(0,229,255,0.16), transparent 58%),
      radial-gradient(620px 360px at 100% 15%, rgba(139,92,246,0.18), transparent 60%),
      linear-gradient(180deg, #070a18 0%, #0a0b1e 100%);
    border-left: 1px solid rgba(255,255,255,0.10);
    box-shadow: -36px 0 90px rgba(0,0,0,0.68);
}}

.about-panel::-webkit-scrollbar {{ width: 7px; }}
.about-panel::-webkit-scrollbar-track {{ background: rgba(255,255,255,0.03); }}
.about-panel::-webkit-scrollbar-thumb {{
    background: linear-gradient(180deg, #00e5ff, #8b5cf6);
    border-radius: 99px;
}}

.about-arrow {{
    position: fixed;
    top: 50%;
    right: min(520px, 94vw);
    transform: translateY(-50%);
    z-index: 10000;
    width: 44px;
    height: 64px;
    background: linear-gradient(180deg, #10142b, #090c1d);
    border: 1px solid rgba(255,255,255,0.12);
    border-right: none;
    border-radius: 14px 0 0 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #67e8f9;
    font-size: 1.4rem;
    font-weight: 700;
    text-decoration: none;
    box-shadow: -10px 0 30px rgba(0,0,0,0.45);
}}

.about-card {{
    padding: 26px;
    border-radius: 28px;
    background:
      radial-gradient(420px 180px at 0% 0%, rgba(0,229,255,0.16), transparent 62%),
      linear-gradient(145deg, rgba(24,28,60,0.92), rgba(13,15,35,0.96));
    border: 1px solid rgba(255,255,255,0.09);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.05), 0 22px 58px rgba(0,0,0,0.32);
    margin-bottom: 18px;
    position: relative;
    overflow: hidden;
}}
.about-card::after {{
    content:"";
    position:absolute;
    inset:-1px;
    background: linear-gradient(135deg, rgba(0,229,255,0.12), transparent 45%, rgba(139,92,246,0.12));
    pointer-events:none;
}}

.about-title {{
    position: relative;
    font-family:'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color:#fff;
    display:flex;
    align-items:center;
    gap:12px;
    margin-bottom: 16px;
}}
.about-title .dot {{
    width:14px;
    height:14px;
    border-radius:50%;
    background: linear-gradient(135deg,#00e5ff,#8b5cf6);
    box-shadow:0 0 18px rgba(0,229,255,.75);
}}
.about-sub {{
    position: relative;
    color:#aeb9cc;
    font-size:.96rem;
    line-height:1.75;
    margin:0;
}}

.about-section {{
    margin-top: 18px;
    padding: 20px;
    border-radius: 24px;
    background: linear-gradient(160deg, rgba(18,21,46,0.74), rgba(10,12,30,0.88));
    border: 1px solid rgba(255,255,255,0.07);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.035);
}}

.about-section h4 {{
    font-family:'JetBrains Mono', monospace;
    font-size:.72rem;
    letter-spacing:.24em;
    text-transform:uppercase;
    color:#67e8f9;
    margin: 0 0 16px 0;
    font-weight: 800;
}}

.about-list {{
    list-style:none;
    padding:0;
    margin:0;
    display:grid;
    gap:12px;
}}

.about-list li {{
    position: relative;
    padding: 15px 16px 15px 38px;
    margin: 0;
    border-radius: 17px;
    background: linear-gradient(160deg, rgba(20,24,52,0.76), rgba(12,14,32,0.9));
    border: 1px solid rgba(255,255,255,0.065);
    color: #cbd5e1;
    font-size: .92rem;
    line-height: 1.62;
    display: block;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.035);
    white-space: normal;
    word-break: keep-all;
    overflow-wrap: normal;
}}

.about-list li::before {{
    content:"";
    position: absolute;
    left: 17px;
    top: 24px;
    width:9px;
    height:9px;
    border-radius:50%;
    background:#00e5ff;
    box-shadow:0 0 14px #00e5ff;
}}

.about-list li b {{
    color:#fff;
    font-weight:800;
}}

.hero {{
    padding: 56px 48px; border-radius: 32px; position: relative; overflow: hidden;
    background:
      radial-gradient(600px 300px at 80% 0%, rgba(0,229,255,0.18), transparent 60%),
      radial-gradient(500px 250px at 0% 100%, rgba(139,92,246,0.18), transparent 60%),
      linear-gradient(145deg, rgba(18,21,46,0.9), rgba(10,12,30,0.95));
    border: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 24px; margin-top: 20px;
}}
.hero::before {{
    content:""; position:absolute; inset:0; opacity:.5;
    background-image:
      linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
    background-size: 44px 44px;
    mask-image: radial-gradient(ellipse at center, black 30%, transparent 80%);
}}
.hero-badge {{
    position:relative; display:inline-flex; align-items:center; gap:8px;
    padding: 8px 14px; border-radius: 999px;
    background: rgba(0,229,255,0.08);
    border: 1px solid rgba(0,229,255,0.25);
    color: #67e8f9; font-size: 11px; font-weight: 700;
    letter-spacing: .18em; text-transform: uppercase; margin-bottom: 22px;
}}
.hero-badge .dot {{ width:8px; height:8px; border-radius:50%; background:#22c55e; box-shadow:0 0 12px #22c55e; animation: pulse 1.6s infinite;}}
@keyframes pulse {{ 0%,100%{{opacity:1;}} 50%{{opacity:.4;}} }}
.hero-title {{
    position:relative; font-family:'Space Grotesk', sans-serif;
    font-size: clamp(2.2rem, 4.6vw, 4rem); font-weight: 700; line-height: 1.05;
    letter-spacing: -0.02em; margin-bottom: 18px;
    display:flex; align-items:center; gap:18px; flex-wrap:wrap;
}}
.hero-title span.grad {{
    background: linear-gradient(90deg,#00e5ff 0%, #8b5cf6 60%, #ec4899 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}
.hero-sub {{ position:relative; max-width: 760px; color:#94a3b8; line-height:1.75; font-size: 1.02rem; }}

.metric-card {{
    padding: 22px; border-radius: 22px; height: 100%;
    background: linear-gradient(160deg, rgba(20,24,52,0.85), rgba(12,14,32,0.9));
    border: 1px solid rgba(255,255,255,0.06);
    transition: .35s cubic-bezier(.2,.8,.2,1);
    position: relative; overflow: hidden;
}}
.metric-card::after{{
    content:""; position:absolute; inset:-1px; border-radius:22px; padding:1px;
    background: linear-gradient(135deg, transparent, rgba(0,229,255,0.4), transparent);
    -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
    -webkit-mask-composite: xor; mask-composite: exclude;
    opacity:0; transition:.35s;
}}
.metric-card:hover {{ transform: translateY(-6px); }}
.metric-card:hover::after {{ opacity:1; }}
.metric-icon {{ font-size: 1.6rem; margin-bottom: 12px; }}
.metric-value {{
    font-family:'Space Grotesk', sans-serif;
    font-size: 1.9rem; font-weight: 700; color:#fff; margin-bottom: 4px;
    background: linear-gradient(180deg,#fff,#94a3b8);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}}
.metric-label {{ color:#94a3b8; font-size: .82rem; letter-spacing:.05em; }}

div[data-testid="stHorizontalBlock"] {{ margin-bottom: 22px; }}
div[data-testid="stHorizontalBlock"] [data-testid="column"] {{
    padding-top: 6px; padding-bottom: 6px;
}}
div[data-testid="stHorizontalBlock"] + div[data-testid="stHorizontalBlock"] {{
    margin-top: 18px;
}}

.section-title {{
    margin: 32px 0 14px; font-size: .76rem; text-transform: uppercase;
    letter-spacing: .2em; color: #67e8f9; font-weight: 700;
    display:flex; align-items:center; gap:10px;
}}
.section-title::before {{
    content:""; width:24px; height:2px; background:linear-gradient(90deg,#00e5ff,transparent);
}}

[data-testid="stForm"] {{
    background: linear-gradient(160deg, rgba(18,21,46,0.85), rgba(10,12,30,0.9));
    border-radius: 28px; border: 1px solid rgba(255,255,255,0.06);
    padding: 30px;
}}
label {{ color:#cbd5e1 !important; text-transform: uppercase;
    font-size:.72rem !important; letter-spacing:.12em; font-weight:700 !important; }}

.stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {{
    background: rgba(5,6,15,0.7) !important; color:#fff !important;
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}}
.stNumberInput input {{ height:52px !important; }}
.stNumberInput input:focus, .stSelectbox div[data-baseweb="select"] > div:focus-within {{
    border-color: #00e5ff !important; box-shadow: 0 0 0 4px rgba(0,229,255,0.12) !important;
}}

.stForm button[kind="primaryFormSubmit"],
.stForm button[kind="secondaryFormSubmit"],
button[data-testid="stBaseButton-primaryFormSubmit"],
button[data-testid="stBaseButton-secondaryFormSubmit"] {{
    width: 100%; height: 64px; border-radius: 18px;
    border: 1px solid rgba(0,229,255,0.35) !important;
    background: linear-gradient(180deg, #0b1530 0%, #060912 100%) !important;
    background-color: #060912 !important;
    color: #ffffff !important;
    font-family:'Space Grotesk', sans-serif !important;
    font-size: 1rem !important; font-weight: 700 !important;
    text-transform: uppercase; letter-spacing: .14em;
    box-shadow:
      inset 0 1px 0 rgba(255,255,255,0.06),
      0 12px 30px rgba(0,229,255,0.18),
      0 0 0 1px rgba(0,229,255,0.15);
    transition: .25s; position: relative; overflow: hidden;
}}
.stForm button[kind="primaryFormSubmit"] *,
button[data-testid="stBaseButton-primaryFormSubmit"] *,
button[data-testid="stBaseButton-secondaryFormSubmit"] * {{
    color: #ffffff !important; fill: #ffffff !important;
}}
.stForm button[kind="primaryFormSubmit"]:hover {{
    transform: translateY(-3px);
    border-color:#00e5ff !important;
    background: linear-gradient(180deg, #0e1a3d 0%, #08101f 100%) !important;
    color:#ffffff !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 18px 50px rgba(0,229,255,0.35);
}}

.result-box {{
    padding: 36px; border-radius: 26px; margin-top: 8px;
    position: relative; overflow: hidden;
}}
.result-box::before {{
    content:""; position:absolute; inset:0; opacity:.35;
    background-image:
      linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
    background-size: 32px 32px;
}}
.result-low {{
    background: radial-gradient(800px 300px at 0% 0%, rgba(255,77,109,0.18), transparent 60%),
                linear-gradient(145deg, rgba(40,12,20,0.9), rgba(15,8,12,0.95));
    border: 1px solid rgba(255,77,109,0.3);
}}
.result-medium {{
    background: radial-gradient(800px 300px at 0% 0%, rgba(255,184,0,0.18), transparent 60%),
                linear-gradient(145deg, rgba(40,30,8,0.9), rgba(15,12,6,0.95));
    border: 1px solid rgba(255,184,0,0.3);
}}
.result-high {{
    background: radial-gradient(800px 300px at 0% 0%, rgba(34,197,94,0.18), transparent 60%),
                linear-gradient(145deg, rgba(8,30,18,0.9), rgba(6,15,10,0.95));
    border: 1px solid rgba(34,197,94,0.3);
}}
.result-tag {{
    position:relative; display:inline-flex; align-items:center; gap:8px;
    padding: 6px 12px; border-radius: 999px;
    background: rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.1);
    font-size:.72rem; text-transform:uppercase; letter-spacing:.18em;
    font-weight:700; margin-bottom: 18px;
}}
.result-title {{
    position:relative; font-family:'Space Grotesk', sans-serif;
    font-size: 2.4rem; font-weight: 700; margin-bottom: 12px; color:#fff;
    letter-spacing: -.01em;
}}
.result-desc {{ position:relative; color:#cbd5e1; line-height: 1.8; max-width: 760px; font-size:1rem; }}

.conf-card {{
    padding: 22px; border-radius: 22px; height: 100%;
    background: linear-gradient(160deg, rgba(20,24,52,0.85), rgba(12,14,32,0.9));
    border: 1px solid rgba(255,255,255,0.06);
    transition:.3s;
}}
.conf-card:hover {{ transform: translateY(-4px); }}
.conf-row {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:14px; }}
.conf-label-text {{
    font-family:'JetBrains Mono', monospace;
    text-transform: uppercase; letter-spacing:.16em; font-weight:700; font-size:.78rem;
}}
.conf-pct {{
    font-family:'Space Grotesk', sans-serif;
    font-size: 1.9rem; font-weight: 700;
}}
.conf-bar {{ height: 8px; background: rgba(255,255,255,0.06); border-radius: 99px; overflow:hidden; }}
.conf-fill {{ height: 100%; border-radius:99px; transition: width 1s ease; }}
.conf-sub {{ color:#64748b; font-size:.78rem; margin-top: 10px; }}
.conf-winner {{ box-shadow: 0 0 0 1px currentColor, 0 12px 40px rgba(0,229,255,0.12); }}

.insight-card {{
    padding: 22px; border-radius: 20px;
    background: linear-gradient(160deg, rgba(18,21,46,0.7), rgba(10,12,30,0.8));
    border: 1px dashed rgba(255,255,255,0.08); height: 100%;
}}
.insight-icon {{ font-size: 1.4rem; margin-bottom: 10px; }}
.insight-title {{ font-weight:700; color:#fff; margin-bottom:6px; }}
.insight-text {{ color:#94a3b8; font-size:.88rem; line-height:1.6; }}

.foot {{ text-align:center; color:#475569; font-size:.78rem; margin-top:40px; letter-spacing:.1em; }}

@media (max-width: 900px) {{
    .nav-row {{ grid-template-columns: 1fr; }}
    .topnav, .about-trigger {{ height: auto; min-height: 72px; }}
    .topnav {{ flex-direction: column; align-items: flex-start; gap: 10px; }}
}}
</style>
{"<script>document.querySelector('.stApp')?.classList.add('about-open');</script>" if st.session_state.show_about else ""}
""", unsafe_allow_html=True)

# =============================================================================
# TOP NAV
# =============================================================================
st.markdown(f"""
<div class="nav-row">
  <div class="topnav">
    <div class="topnav-brand">
      <div class="logo"></div>
      <div class="name">MUHAMMED SWALIH</div>
    </div>
    <div class="topnav-tag">Built by curiosity · Driven by intelligence</div>
  </div>
  <a class="about-trigger" href="{about_href}" target="_self">ⓘ About</a>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# ABOUT SLIDE-OVER
# =============================================================================
if st.session_state.show_about:
    st.markdown("""
    <div class="about-overlay"><a class="full" href="?about=0" target="_self"></a></div>
    <a class="about-arrow" href="?about=0" target="_self">›</a>
    <div class="about-panel">
      <div class="about-card">
        <div class="about-title"><span class="dot"></span> About</div>
        <p class="about-sub">
          A machine learning system for predicting player engagement using
          behavioural &amp; gameplay metrics — turning raw telemetry into
          retention strategy.
        </p>
      </div>

      <div class="about-section">
        <h4>How it works</h4>
        <ul class="about-list">
          <li>Collects <b>11 player features</b> across gameplay, demographics &amp; preferences.</li>
          <li>Feeds them into an <b>XGBoost</b> classification pipeline.</li>
          <li>Returns a <b>3-class engagement label</b> with confidence.</li>
        </ul>
      </div>

      <div class="about-section">
        <h4>Signals We Read</h4>
        <ul class="about-list">
          <li><b>Demographics</b> — age, gender, location</li>
          <li><b>Gameplay</b> — play time, sessions, session length</li>
          <li><b>Progression</b> — level, achievements unlocked</li>
          <li><b>Preferences</b> — genre, difficulty, purchases</li>
        </ul>
      </div>

      <div class="about-section">
        <h4>Engagement Tiers</h4>
        <ul class="about-list">
          <li><b>Low</b> — casual players with minimal engagement</li>
          <li><b>Medium</b> — regular players with steady engagement</li>
          <li><b>High</b> — dedicated players, deeply engaged</li>
        </ul>
      </div>

      <div class="about-section">
        <h4>Tech stack</h4>
        <ul class="about-list">
          <li>Python · Pandas · Scikit-learn · XGBoost · Streamlit</li>
        </ul>
      </div>

      <div class="about-section">
        <h4>Use cases</h4>
        <ul class="about-list">
          <li>Player retention strategy</li>
          <li>Personalised in-game offers</li>
          <li>Churn risk early warning</li>
        </ul>
      </div>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# HERO
# =============================================================================
st.markdown("""
<div class="hero">
  <div class="hero-badge"><span class="dot"></span> Gaming Engagement Level Prediction · Live</div>
  <div class="hero-title"><span class="grad">Predict the next move of every player.</span></div>
  <p class="hero-sub">
    Predict player engagement levels using advanced machine learning based on gameplay behavior
    and player characteristics. Get instant insights with confidence scores!
  </p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# METRIC CARDS
# =============================================================================
cards = [
    ("🎯", "3-Class", "Engagement Tiers"),
    ("📊", "11", "Behavioural Signals"),
    ("🧠", "XGBoost", "Gradient Boosted Trees"),
    ("🛡️", "92%+", "Validation Accuracy"),
]
cols = st.columns(4, gap="medium")
for col, (icon, value, label) in zip(cols, cards):
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-icon">{icon}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# FORM
# =============================================================================
st.markdown('<div class="section-title">Player Telemetry Input</div>', unsafe_allow_html=True)

with st.form("prediction_form"):
    c1, c2 = st.columns(2, gap="large")
    with c1:
        age = st.number_input("Age", 10, 100, 25)
        play_time = st.number_input("Total Play Time (Hours)", 0.0, 10000.0, 120.0)
        sessions_per_week = st.number_input("Sessions Per Week", 0, 50, 7)
        avg_session_duration = st.number_input("Average Session Duration (Min)", 1, 480, 60)
        player_level = st.number_input("Player Level", 1, 200, 15)
        achievements = st.number_input("Achievements Unlocked", 0, 500, 10)

    with c2:
        in_game_purchases = st.selectbox("In-Game Purchases", ["No", "Yes"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        location = st.selectbox("Location", ["USA", "Europe", "Asia", "Other"])
        game_genre = st.selectbox("Game Genre", ["Action", "RPG", "Sports", "Strategy", "Simulation"])
        game_difficulty = st.selectbox("Game Difficulty", ["Easy", "Medium", "Hard"])

    predict = st.form_submit_button("⚡  Run Engagement Prediction", use_container_width=True)

# =============================================================================
# PREDICTION OUTPUT
# =============================================================================
if predict:
    input_df = pd.DataFrame([{
        "Age": age, "PlayTimeHours": play_time, "SessionsPerWeek": sessions_per_week,
        "AvgSessionDurationMinutes": avg_session_duration, "PlayerLevel": player_level,
        "AchievementsUnlocked": achievements,
        "InGamePurchases": 1 if in_game_purchases == "Yes" else 0,
        "Gender": gender, "Location": location,
        "GameGenre": game_genre, "GameDifficulty": game_difficulty,
    }])

    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    labels = ["Low", "Medium", "High"]
    final_label = labels[prediction]
    confidence = probabilities[prediction]

    result_map = {
        "Low":    {"class":"result-low","tag":"⚠ LOW ENGAGEMENT",
                   "title":"At-Risk Player",
                   "desc":"This player shows weakening session patterns. Consider re-engagement campaigns, personalised rewards, or onboarding nudges to recover activity."},
        "Medium": {"class":"result-medium","tag":"📈 MODERATE ENGAGEMENT",
                   "title":"Steady Player",
                   "desc":"Balanced behaviour with stable retention. Ideal target for upsell, social features and tier-progression incentives to elevate to high engagement."},
        "High":   {"class":"result-high","tag":"🏆 HIGHLY ENGAGED",
                   "title":"Power Player",
                   "desc":"Top-tier engagement. Reward loyalty, unlock community/creator tools, and use this segment for early access, beta tests and ambassador programmes."},
    }
    r = result_map[final_label]

    st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="result-box {r['class']}">
      <div class="result-tag">{r['tag']} · {confidence:.1%} confidence</div>
      <div class="result-title">{r['title']}</div>
      <div class="result-desc">{r['desc']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Confidence Distribution</div>', unsafe_allow_html=True)
    colors = {"Low": "#ff4d6d", "Medium": "#ffb800", "High": "#22c55e"}
    cols = st.columns(3, gap="medium")
    for col, label, prob in zip(cols, labels, probabilities):
        winner = "conf-winner" if label == final_label else ""
        with col:
            st.markdown(f"""
            <div class="conf-card {winner}" style="color:{colors[label]}">
              <div class="conf-row">
                <div class="conf-label-text" style="color:{colors[label]}">{label}</div>
                <div class="conf-pct" style="color:{colors[label]}">{prob:.1%}</div>
              </div>
              <div class="conf-bar">
                <div class="conf-fill" style="width:{prob*100:.1f}%; background:{colors[label]}"></div>
              </div>
              <div class="conf-sub">Probability the player belongs to this engagement tier.</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Roadmap & Future Enhancements</div>', unsafe_allow_html=True)
    insights = [
        ("🔮", "SHAP Explainability", "Show exactly which features pushed this player into their predicted tier — boosts trust and decisioning."),
        ("📡", "Real-time Telemetry", "Stream live game events via Kafka / WebSockets and re-score players continuously."),
        ("🧪", "A/B Recommendation Engine", "Auto-suggest retention actions (offers, missions, nudges) per engagement tier."),
        ("📈", "Cohort Dashboard", "Visualise tier migration over time — who moved from Medium → High after a campaign."),
        ("🤖", "LLM Player Coach", "Generate natural-language summaries: 'Why is this player likely to churn?'"),
        ("🛒", "Revenue & LTV Model", "Pair engagement with predicted lifetime value to prioritise marketing spend."),
    ]
    rows = [insights[:3], insights[3:]]
    for row in rows:
        cols = st.columns(3, gap="medium")
        for col, (ic, t, d) in zip(cols, row):
            with col:
                st.markdown(f"""
                <div class="insight-card">
                  <div class="insight-icon">{ic}</div>
                  <div class="insight-title">{t}</div>
                  <div class="insight-text">{d}</div>
                </div>
                """, unsafe_allow_html=True)

st.markdown('<div class="foot">MUHAMMED SWALIH · Crafted with XGBoost · © 2025</div>', unsafe_allow_html=True)

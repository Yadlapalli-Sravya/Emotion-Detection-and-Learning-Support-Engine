import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
from datetime import datetime

# Import prediction logic
from src.predict import predict_emotion

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="Emotion Detection & Learning Support Engine",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.switch_page("login.py")


# ==========================================================
# SESSION STATE INITIALIZATION
# ==========================================================
if "history" not in st.session_state:
    st.session_state.history = []

if "interaction_count" not in st.session_state:
    st.session_state.interaction_count = 0

if "student_text" not in st.session_state:
    st.session_state.student_text = ""

if "auto_trigger" not in st.session_state:
    st.session_state.auto_trigger = False

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ==========================================================
# ANALYTICS METRICS COMPUTATION
# ==========================================================
def get_analytics_metrics():
    if not st.session_state.history:
        return 0, 0.0, "N/A"
    
    total = len(st.session_state.history)
    avg_conf = np.mean([item["confidence"] for item in st.session_state.history])
    
    # Calculate dominant emotion
    emotions = [item["emotion"] for item in st.session_state.history]
    dominant = max(set(emotions), key=emotions.count)
    
    return total, avg_conf, dominant

# ==========================================================
# MIXED EMOTION DETECTION
# ==========================================================

def get_mixed_emotions(probabilities, class_names, threshold=0.15):

    emotion_scores = dict(zip(class_names, probabilities))

    sorted_emotions = sorted(
        emotion_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    primary = sorted_emotions[0]

    mixed = [primary]

    for emotion, score in sorted_emotions[1:]:

        if score >= threshold:
            mixed.append((emotion, score))

    return mixed   

# ==========================================================
# CUSTOM STYLING (CSS)
# ==========================================================
# Light & Dark mode support using Streamlit theme variables
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

/* Global Reset & Typography */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: var(--text-color);
}

/* Background gradient for main app area */
.stApp {
    background: linear-gradient(135deg, var(--background-color) 0%, var(--secondary-background-color) 100%);
}

/* Glassmorphism containers */
.glass-container {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.15);
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.02);
    margin-bottom: 20px;
}

/* Hero Banner Container */
.hero-banner {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.15);
    border-bottom: 4px solid #4F46E5;
    border-radius: 16px;
    padding: 25px;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.02);
}

/* Sidebar Custom Styling */
[data-testid="stSidebar"] {
    background-color: var(--secondary-background-color);
    border-right: 1px solid rgba(128, 128, 128, 0.15);
}

/* Customize Streamlit Tabs */
button[data-baseweb="tab"] {
    font-size: 15px !important;
    font-weight: 600 !important;
    color: #64748B !important;
    padding: 12px 24px !important;
    transition: all 0.2s ease-in-out !important;
    background-color: transparent !important;
    border: none !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #4F46E5 !important;
    border-bottom: 3px solid #4F46E5 !important;
}

/* Customize Primary Streamlit Buttons */
.stButton>button {
    background: linear-gradient(135deg, #4F46E5 0%, #3730A3 100%);
    color: white !important;
    border-radius: 10px;
    border: none;
    font-weight: 600;
    transition: all 0.2s ease-in-out;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.15);
}

.stButton>button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(79, 70, 229, 0.25);
    border: none !important;
}

/* Textarea custom style */
textarea {
    border-radius: 12px !important;
    border: 1px solid rgba(128, 128, 128, 0.2) !important;
    font-family: inherit !important;
    font-size: 15px !important;
    background-color: var(--background-color) !important;
    color: var(--text-color) !important;
    transition: all 0.2s ease !important;
}

textarea:focus {
    border-color: #4F46E5 !important;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.12) !important;
}

/* Status badge */
.status-badge {
    background-color: rgba(16, 185, 129, 0.15);
    color: #10B981;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

/* Cards UI */
.metric-card {
    background: var(--secondary-background-color);
    border-radius: 12px;
    padding: 20px;
    border: 1px solid rgba(128, 128, 128, 0.15);
    text-align: center;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.01);
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: var(--text-color);
}

.metric-label {
    font-size: 12px;
    color: var(--text-color);
    opacity: 0.7;
    text-transform: uppercase;
    font-weight: 600;
    margin-top: 6px;
    letter-spacing: 0.05em;
}

.sidebar-metric {
    background: var(--background-color);
    padding: 14px;
    border-radius: 10px;
    border: 1px solid rgba(128, 128, 128, 0.15);
    margin-bottom: 10px;
    text-align: center;
}

.sidebar-metric-value {
    font-size: 22px;
    font-weight: 700;
    color: var(--text-color);
}

.sidebar-metric-label {
    font-size: 10px;
    color: var(--text-color);
    opacity: 0.7;
    text-transform: uppercase;
    font-weight: 600;
    margin-top: 2px;
    letter-spacing: 0.05em;
}

/* Custom Card Classes for Light/Dark mode compatibility */
.advice-card-custom {
    background-color: var(--background-color);
    border: 1px solid rgba(128, 128, 128, 0.15);
    padding: 28px;
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.02);
    color: var(--text-color);
}

.advice-quote-box {
    font-size: 15px;
    line-height: 1.6;
    color: var(--text-color);
    margin: 0 0 20px 0;
    background-color: var(--secondary-background-color);
    padding: 16px;
    border-radius: 12px;
    border: 1px solid rgba(128, 128, 128, 0.1);
    font-style: italic;
}

.advice-strategy-box {
    background-color: var(--secondary-background-color);
    border-radius: 12px;
    padding: 20px;
    border: 1px dashed rgba(128, 128, 128, 0.25);
}

.full-session-log-card {
    background-color: var(--background-color);
    border: 1px solid rgba(128, 128, 128, 0.15);
    padding: 12px 16px;
    border-radius: 8px;
    margin-bottom: 8px;
    font-size: 14px;
    color: var(--text-color);
}

/* Header typography */
.hero-title {
    font-size: 34px;
    font-weight: 800;
    background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    font-size: 15px;
    color: var(--text-color);
    opacity: 0.85;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# EMOTION CONFIGURATIONS & STYLES
# ==========================================================
# Light/Dark mode neutral translucent glass colors
emotion_styles = {
    "Confident": {
        "emoji": "😎",
        "color": "#10B981",
        "bg_color": "rgba(16, 185, 129, 0.12)",
        "border_color": "rgba(16, 185, 129, 0.22)",
        "text_color": "#10B981",
        "description": "You are feeling self-assured, positive, and capable of solving your learning objectives.",
        "bullets": [
            "<b>Deepen Mastery</b>: Try explaining the concept to a classmate or writing a brief summary. Explaining is the highest form of learning.",
            "<b>Extend Scope</b>: Challenge yourself with edge-cases, code optimization, or advanced projects based on the topic.",
            "<b>Build & Deploy</b>: Apply the knowledge to a sandbox project to ensure long-term retrieval."
        ]
    },
    "Confused": {
        "emoji": "😕",
        "color": "#F59E0B",
        "bg_color": "rgba(245, 158, 11, 0.12)",
        "border_color": "rgba(245, 158, 11, 0.22)",
        "text_color": "#F59E0B",
        "description": "You are experiencing uncertainty, lack of clarity, or difficulty understanding a specific concept.",
        "bullets": [
            "<b>Feynman Technique</b>: Write down the concept in the simplest terms possible, as if teaching a child. Identify where your explanation breaks down.",
            "<b>Visual Mapping</b>: Sketch a diagram or flowchart of the components to trace how data flow or logic relates.",
            "<b>Micro-Steps</b>: Divide the confusing concept into independent 5-minute segments and read them sequentially."
        ]
    },
    "Curious": {
        "emoji": "🤔",
        "color": "#3B82F6",
        "bg_color": "rgba(59, 130, 246, 0.12)",
        "border_color": "rgba(59, 130, 246, 0.22)",
        "text_color": "#3B82F6",
        "description": "You are showing an eager desire to learn, explore, and ask deeper architectural questions.",
        "bullets": [
            "<b>Deep-Dive Resources</b>: Look up standard technical specifications, official documentations, or research papers on the topic.",
            "<b>Experimental Sandbox</b>: Write short scripts to test 'what-if' scenarios, changing parameters to inspect outputs.",
            "<b>Collaborative Discussion</b>: Share your findings or questions on dev forums or local study channels to hear alternative perspectives."
        ]
    },
    "Frustrated": {
        "emoji": "😣",
        "color": "#EF4444",
        "bg_color": "rgba(239, 68, 68, 0.12)",
        "border_color": "rgba(239, 68, 68, 0.22)",
        "text_color": "#EF4444",
        "description": "You are feeling stuck, annoyed, or discouraged by a bug or high learning curve.",
        "bullets": [
            "<b>Take a Pomodoro Break</b>: Step away from your computer for 10 minutes. A relaxed brain resolves bugs much faster.",
            "<b>Rubber Duck Debugging</b>: Read your code or problem statement step-by-step to an inanimate object. This forces structured processing.",
            "<b>Isolate the Issue</b>: Temporarily comment out external dependencies. Create a minimal reproducible example of the failure."
        ]
    }
}

# ==========================================================
# SIDEBAR
# ==========================================================
with st.sidebar:
    st.markdown("## 📊 Engine Control")
    
    # 1. Model status badge
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <span style="font-size: 13px; opacity: 0.8;">Model Engine:</span><br>
        <span class="status-badge">✅ BiLSTM Loaded</span>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Session KPIs
    st.markdown("### 📈 Session Metrics")
    total, avg_conf, dominant = get_analytics_metrics()
    
    # Select styling for dominant emotion
    dom_style = emotion_styles.get(dominant, {"emoji": "🎓"})
    dom_display = f"{dom_style['emoji']} {dominant}" if dominant != "N/A" else "N/A"
    
    st.markdown(f"""
    <div class="sidebar-metric">
        <div class="sidebar-metric-value">{total}</div>
        <div class="sidebar-metric-label">Analyses Run</div>
    </div>
    <div class="sidebar-metric">
        <div class="sidebar-metric-value">{avg_conf * 100:.1f}%</div>
        <div class="sidebar-metric-label">Avg Confidence</div>
    </div>
    <div class="sidebar-metric">
        <div class="sidebar-metric-value">{dom_display}</div>
        <div class="sidebar-metric-label">Dominant State</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 3. History Actions
    st.markdown("### ⚙️ Control Actions")
    if st.button("🗑 Reset Session History", use_container_width=True, key="btn_reset_session"):
        st.session_state.history = []
        st.session_state.interaction_count = 0
        st.session_state.last_result = None
        st.toast("Session history reset successfully!")
        
        # Support rerun across streamlist versions
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()
    st.markdown("---")

    st.write(f"👤 Logged in as: {st.session_state.user}")

    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.user = None
        st.switch_page("login.py")

# ==========================================================
# MAIN APP INTERFACE
# ==========================================================
# Styled Hero Header
st.markdown("""
<div class="hero-banner">
    <h1 class="hero-title">🎓 Emotion Detection & Learning Support Engine</h1>
    <p class="hero-subtitle">
        Understand your learning obstacles, analyze sentiment with our BiLSTM classifier, and access targeted study strategies.
    </p>
</div>
""", unsafe_allow_html=True)

# Define Tabs
tab1, tab2, tab3 = st.tabs([
    "🎯 Analyze Emotion", 
    "📊 Session History & Analytics", 
    "🛠️ Neural Network Diagnostics"
])

# Template Callback
def load_example(text):
    st.session_state.student_text = text
    st.session_state.auto_trigger = True

# ==========================================================
# TAB 1: INTERACTIVE EMOTION ANALYSIS
# ==========================================================
with tab1:
    col_in, col_temp = st.columns([1.5, 1])
    
    with col_in:
        st.markdown("### ✍️ Input Your Challenge")
        field = st.selectbox(
            "📚 Select Study Field",
            [
                "Computer Science",
                "Artificial Intelligence",
                "Machine Learning",
                "Data Science",
                "Mathematics",
                "Physics",
                "Chemistry",
                "Biology",
                "General Academic"
            ],
            key="study_field"
        )
        
        student_text = st.text_area(
            "Describe what you are working on, what is causing difficulty, or how you feel about it:",
            key="student_text",
            height=140,
            placeholder="Example: I'm trying to learn recursion, but my functions keep hitting stack overflows. I'm completely stuck..."
        )
        
        submit_clicked = st.button("🚀 Analyze Emotion", use_container_width=True, key="btn_analyze_sentiment")
        
        st.caption("💡 *Note: The classifying model is trained on student expressions. Including emotion phrases like 'I feel confused' or 'I am frustrated' helps the NLP engine deliver higher accuracy predictions.*")
        
    with col_temp:
        st.markdown("### 💡 Click to Test Templates")
        st.caption("Select a preset challenge matching target emotional states:")
        
        # Style layout for preset buttons using aligned expressive phrases
        st.button(
            "😕 Confused - Recursion base cases",
            use_container_width=True,
            on_click=load_example,
            args=("I feel confused and nervous because I don't understand how recursion works. I'm afraid I'll fail the exam.",),
            key="tpl_confused"
        )
        
        st.button(
            "🤔 Curious - Focal Loss vs Cross-Entropy",
            use_container_width=True,
            on_click=load_example,
            args=("I am curious and shocked by how focal loss works compared to standard cross-entropy. I want to learn more about this amazing technique.",),
            key="tpl_curious"
        )
        
        st.button(
            "😣 Frustrated - 3-hour segfault debug",
            use_container_width=True,
            on_click=load_example,
            args=("I feel angry and frustrated. I've been trying to fix this core dump in C++ for three hours. The pointers look correct but it keeps crashing. I hate it.",),
            key="tpl_frustrated"
        )
        
        st.button(
            "😎 Confident - Successful validation run",
            use_container_width=True,
            on_click=load_example,
            args=("I feel happy and confident because I successfully implemented the tokenization pipeline and hit 92% validation accuracy!",),
            key="tpl_confident"
        )

    # Trigger logic on submit or template auto trigger
    if submit_clicked or st.session_state.auto_trigger:
        st.session_state.auto_trigger = False # Reset trigger
        
        if st.session_state.student_text.strip() == "":
            st.warning("⚠️ Please input text or select a template before running the analysis.")
        else:
            with st.spinner("🧠 BiLSTM model is evaluating text features..."):
                try:
                    emotion, advice, confidence, probabilities, class_names = predict_emotion(st.session_state.student_text)
                    mixed_emotions = get_mixed_emotions(
                        probabilities,
                        class_names,
                        threshold=0.15
                )
                    st.session_state.interaction_count += 1
                    
                    # Log to history
                    st.session_state.history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "field": field,
                        "text": st.session_state.student_text,
                        "emotion": emotion,
                        "confidence": float(confidence)
                    })
                    
                    # Store last result
                    st.session_state.last_result = {
                        "emotion": emotion,
                        "mixed_emotions": mixed_emotions,
                        "advice": advice,
                        "confidence": float(confidence),
                        "probabilities": [float(p) for p in probabilities],
                        "class_names": list(class_names),
                        "field": field,
                        "text": st.session_state.student_text
                    }
                    
                    st.toast("Emotion analyzed!", icon="🧠")
                    
                except Exception as e:
                    st.error(f"Prediction failed: {e}. Please check model files.")

    # Display results if available
    if st.session_state.last_result is not None:
        res = st.session_state.last_result
        emotion = res["emotion"]
        mixed_emotions = res["mixed_emotions"]
        advice = res["advice"]
        confidence = res["confidence"]
        probabilities = res["probabilities"]
        class_names = res["class_names"]
        field_selected = res["field"]
        
        st.markdown("<br><hr style='border-top: 1px solid rgba(128, 128, 128, 0.15);'>", unsafe_allow_html=True)
        st.markdown("## 🔍 Analysis Output")
        
        col_res1, col_res2 = st.columns([1, 1.2])
        
        with col_res1:
            # Get style configurations
            style = emotion_styles.get(emotion, {
                "emoji": "🙂",
                "color": "#4F46E5",
                "bg_color": "rgba(79, 70, 229, 0.1)",
                "border_color": "rgba(79, 70, 229, 0.2)",
                "text_color": "#4F46E5",
                "description": "General learning emotional state detected.",
                "bullets": ["Continue exploring standard solutions."]
            })
            
            # HTML Emotion Badge Card (Legible in light and dark mode)
            if len(mixed_emotions) > 1:
                emotion_title = " + ".join(
                 [e[0] for e in mixed_emotions]
                )
            else:
                 emotion_title = emotion
            st.markdown(f"""
            <div style="
                background-color: {style['bg_color']};
                border: 1px solid {style['border_color']};
                border-left: 8px solid {style['color']};
                padding: 24px;
                border-radius: 16px;
                color: var(--text-color);
                display: flex;
                align-items: center;
                gap: 20px;
                margin-bottom: 24px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.02);
            ">
                <div style="font-size: 52px; line-height: 1;">{style['emoji']}</div>
                <div>
                    <h4 style="margin: 0; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.8;">Detected State</h4>
                    <h2 style="margin: 4px 0 8px 0; font-size: 26px; font-weight: 700; color: {style['text_color']};">{emotion_title}</h2>
                    <p style="margin: 0; font-size: 14px; opacity: 0.9; line-height: 1.45;">{style['description']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress bar for confidence
            st.markdown("### 📊 Classification Confidence")
            st.progress(confidence)
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: -10px; margin-bottom: 24px;">
                <span style="font-size: 13px; opacity: 0.7;">Classifier Probability Score</span>
                <span style="font-size: 15px; font-weight: 700; color: {style['text_color']}; background-color: {style['bg_color']}; padding: 2px 10px; border-radius: 12px; border: 1px solid {style['border_color']};">{confidence*100:.2f}%</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Altair Probability Chart (Adapts to Light/Dark background theme)
            st.markdown("### 📈 State Probabilities")
            
            df_prob = pd.DataFrame({
                "State": class_names,
                "Probability": probabilities
            })
            
            color_scale = alt.Scale(
                domain=["Confident", "Confused", "Curious", "Frustrated"],
                range=["#10B981", "#F59E0B", "#3B82F6", "#EF4444"]
            )
            
            prob_chart = alt.Chart(df_prob).mark_bar(
                cornerRadiusTopRight=6,
                cornerRadiusBottomRight=6,
                height=22
            ).encode(
                x=alt.X("Probability:Q", title="Probability Score", axis=alt.Axis(format="%", grid=False)),
                y=alt.Y("State:N", title=None, sort="-x", axis=alt.Axis(labelFontSize=12, tickSize=0, labelFont="Plus Jakarta Sans")),
                color=alt.Color("State:N", scale=color_scale, legend=None),
                tooltip=[alt.Tooltip("State:N"), alt.Tooltip("Probability:Q", format=".2%")]
            ).properties(
                height=180
            ).configure_view(
                strokeWidth=0
            )
            
            st.altair_chart(prob_chart, use_container_width=True)
            st.markdown("### 🎭 Mixed Emotion Detection")

            for emotion_name, score in mixed_emotions:

                st.progress(float(score))

                st.write(f"**{emotion_name}** — {score*100:.2f}%")

        with col_res2:
            # HTML Advice and Bullet Strategies Card (Perfect Contrast in Light & Dark Mode)
            bullets_html = "".join([f"<li style='margin-bottom: 10px;'>{b}</li>" for b in style["bullets"]])
            
            st.markdown(f"""
            <div class="advice-card-custom" style="border-left: 8px solid {style['color']};">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 16px;">
                    <span style="background-color: {style['bg_color']}; padding: 6px 12px; border-radius: 20px; font-size: 11px; font-weight: 700; color: {style['text_color']}; border: 1px solid {style['border_color']}; text-transform: uppercase; letter-spacing: 0.05em;">
                        🤖 AI Learning Assistant
                    </span>
                    <span style="font-size: 13px; opacity: 0.8;">Field: <b>{field_selected}</b></span>
                </div>
                <h3 style="margin: 0 0 12px 0; font-size: 20px; font-weight: 700; color: var(--text-color);">Personalized Recommendation</h3>
                <div class="advice-quote-box">
                    " {advice} "
                </div>
                <div class="advice-strategy-box">
                    <h4 style="margin: 0 0 10px 0; font-size: 14px; font-weight: 700; color: var(--text-color);">
                        💡 Recommended Study Strategies:
                    </h4>
                    <ul style="margin: 0; padding-left: 20px; font-size: 14px; color: var(--text-color); opacity: 0.9; line-height: 1.55;">
                        {bullets_html}
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ==========================================================
# TAB 2: SESSION HISTORY & ANALYTICS
# ==========================================================
with tab2:
    total, avg_conf, dominant = get_analytics_metrics()
    
    if total == 0:
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px; background-color: var(--secondary-background-color); border-radius: 16px; border: 1px dashed rgba(128, 128, 128, 0.25); margin-top: 10px;">
            <div style="font-size: 64px; margin-bottom: 20px;">📊</div>
            <h3 style="margin: 0 0 8px 0; color: var(--text-color); font-weight: 700;">No Analytics Available Yet</h3>
            <p style="margin: 0; color: var(--text-color); opacity: 0.7; font-size: 14px;">Submit challenges in the <b>Analyze Emotion</b> tab to accumulate session history and view emotional patterns.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("### 📈 Session Summary Metrics")
        
        # Row of KPI Cards
        col_k1, col_k2, col_k3 = st.columns(3)
        
        with col_k1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total}</div>
                <div class="metric-label">Analyses Logged</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_k2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_conf * 100:.1f}%</div>
                <div class="metric-label">Average Confidence</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_k3:
            dom_style = emotion_styles.get(dominant, {"emoji": "🎓", "color": "#4F46E5"})
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid {dom_style['color']};">
                <div class="metric-value">{dom_style['emoji']} {dominant}</div>
                <div class="metric-label">Dominant Emotion</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Charts Grid
        col_chart1, col_chart2 = st.columns([1.3, 1])
        
        with col_chart1:
            st.markdown("### 📈 Sentiment Journey Index")
            
            # Map states to linear sentiment value
            sentiment_map = {
                "Confident": 1.0,
                "Curious": 0.5,
                "Confused": -0.2,
                "Frustrated": -0.8
            }
            
            hist_list = []
            for idx, item in enumerate(st.session_state.history):
                hist_list.append({
                    "Index": idx + 1,
                    "Challenge": item["text"][:35] + "..." if len(item["text"]) > 35 else item["text"],
                    "Emotion": item["emotion"],
                    "Confidence": item["confidence"],
                    "Sentiment Index": sentiment_map.get(item["emotion"], 0.0),
                    "Timestamp": item["timestamp"]
                })
                
            df_hist = pd.DataFrame(hist_list)
            
            # Line chart showing emotional trend (Dynamic Light & Dark Theme Formatting)
            line_chart = alt.Chart(df_hist).mark_line(
                color="#4F46E5",
                strokeWidth=3,
                point=alt.OverlayMarkDef(color="#4F46E5", size=50, filled=True)
            ).encode(
                x=alt.X("Index:Q", title="Sequence of Queries", axis=alt.Axis(tickMinStep=1)),
                y=alt.Y("Sentiment Index:Q", title="Sentiment Value", scale=alt.Scale(domain=[-1.1, 1.1])),
                tooltip=["Index:Q", "Emotion:N", "Confidence:Q", "Timestamp:N", "Challenge:N"]
            ).properties(
                height=260
            ).configure_view(
                strokeWidth=0
            )
            
            st.altair_chart(line_chart, use_container_width=True)
            st.caption("ℹ️ Sentiment Journey Index maps emotions to a spectrum between -1.0 (Struggling/Frustrated) and +1.0 (Confident).")
            
        with col_chart2:
            st.markdown("### 📋 Recent Analyses Logs")
            
            # Render log cards for recent actions (Perfect Light/Dark colors)
            for idx, item in enumerate(reversed(st.session_state.history)):
                if idx >= 5: # Limit list view to 5
                    break
                istyle = emotion_styles.get(item["emotion"], {"emoji": "🙂", "color": "#4F46E5", "bg_color": "rgba(79, 70, 229, 0.1)", "border_color": "rgba(79, 70, 229, 0.2)", "text_color": "#4F46E5"})
                
                st.markdown(f"""
                <div class="full-session-log-card" style="border-left: 5px solid {istyle['color']}; border-color: {istyle['border_color']};">
                    <div style="display: flex; justify-content: space-between; font-weight: 600; color: {istyle['text_color']}; margin-bottom: 4px;">
                        <span>{istyle['emoji']} {item['emotion']} ({item['confidence']*100:.1f}%)</span>
                        <span style="font-size: 11px; color: var(--text-color); opacity: 0.6; font-weight: 400;">{item['timestamp'].split(' ')[1]}</span>
                    </div>
                    <div style="color: var(--text-color); opacity: 0.9; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                        "{item['text']}"
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        with st.expander("📂 View Complete Session Logs Table"):
            full_df = pd.DataFrame(st.session_state.history)
            st.dataframe(
                full_df[["timestamp", "field", "text", "emotion", "confidence"]].rename(
                    columns={
                        "timestamp": "Timestamp",
                        "field": "Study Field",
                        "text": "Challenge Text",
                        "emotion": "Emotion Predicted",
                        "confidence": "Probability Score"
                    }
                ),
                use_container_width=True
            )

# ==========================================================
# TAB 3: NEURAL NETWORK DIAGNOSTICS
# ==========================================================
with tab3:
    st.markdown("### 🧠 Model Architecture & Pipeline Diagnostics")
    
    col_d1, col_d2 = st.columns([1, 1.1])
    
    with col_d1:
        st.markdown("""
        <div style="background: var(--background-color); border: 1px solid rgba(128, 128, 128, 0.15); padding: 24px; border-radius: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.01); height: 100%; color: var(--text-color);">
            <h4 style="margin-top: 0; color: var(--text-color); font-weight: 700; font-size: 18px;">⚙️ BiLSTM Architecture</h4>
            <p style="font-size: 14px; opacity: 0.85; line-height: 1.55;">
                The classifying engine utilizes a <b>Bidirectional Long Short-Term Memory (BiLSTM)</b> neural network to evaluate features. By analyzing sequences forward and backward, the model captures word relationships in both contexts.
            </p>
            <hr style="border-top: 1px solid rgba(128, 128, 128, 0.15); margin: 15px 0;">
            <table style="width: 100%; font-size: 13px; color: var(--text-color); border-collapse: collapse;">
                <tr style="border-bottom: 1px solid rgba(128, 128, 128, 0.15); height: 32px; opacity: 0.9;">
                    <td style="font-weight: 600; width: 140px;">Vocabulary Limit</td>
                    <td>30,000 unique tokens</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(128, 128, 128, 0.15); height: 32px; opacity: 0.9;">
                    <td style="font-weight: 600;">Max Length</td>
                    <td>80 words (with post-padding)</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(128, 128, 128, 0.15); height: 32px; opacity: 0.9;">
                    <td style="font-weight: 600;">Embedding Dimension</td>
                    <td>128 dense dimensions</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(128, 128, 128, 0.15); height: 32px; opacity: 0.9;">
                    <td style="font-weight: 600;">LSTM Hidden Layer</td>
                    <td>128 units (dropout=0.2, recurrent=0.2)</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(128, 128, 128, 0.15); height: 32px; opacity: 0.9;">
                    <td style="font-weight: 600;">Dense Layer</td>
                    <td>128 hidden units (ReLU activation)</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(128, 128, 128, 0.15); height: 32px; opacity: 0.9;">
                    <td style="font-weight: 600;">Regularization</td>
                    <td>Dropout Layer (rate=0.3)</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(128, 128, 128, 0.15); height: 32px; opacity: 0.9;">
                    <td style="font-weight: 600;">Output classes</td>
                    <td>4 states (Softmax)</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
    with col_d2:
        st.markdown("""
        <div style="background: var(--background-color); border: 1px solid rgba(128, 128, 128, 0.15); padding: 24px; border-radius: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.01); height: 100%; color: var(--text-color);">
            <h4 style="margin-top: 0; color: var(--text-color); font-weight: 700; font-size: 18px;">🎯 Loss Objective Function: Focal Loss</h4>
            <p style="font-size: 14px; opacity: 0.85; line-height: 1.55;">
                Standard cross-entropy loss degrades in performance when training data suffers from class imbalances (e.g. curiosity text is less frequent than frustrated descriptions). 
                The training pipeline resolves this by implementing a custom multi-class <b>Focal Loss</b> equation:
            </p>
            <code style="display: block; background-color: var(--secondary-background-color); padding: 12px; border-radius: 8px; font-size: 13px; color: var(--text-color); font-family: monospace; border: 1px solid rgba(128, 128, 128, 0.15); margin-bottom: 12px; font-weight: bold;">
                FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
            </code>
            <p style="font-size: 13px; opacity: 0.8; line-height: 1.5;">
                By multiplying cross-entropy by the scaling factor <code>(1 - p_t)^γ</code> (where γ=2.0), the model dynamically discounts gradients from simple, common phrases during training, directing model weights to focus on distinguishing rare or complex text cues.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Metrics & Performance Validation")
    
    col_img1, col_img2 = st.columns(2)
    
    with col_img1:
        if os.path.exists("models/confusion_matrix.png"):
            st.image("models/confusion_matrix.png", caption="Model Confusion Matrix (Validation Split)", use_column_width=True)
        else:
            st.info("Confusion matrix visualization currently unavailable.")
            
    with col_img2:
        if os.path.exists("models/domain_adaptive_loss.png"):
            st.image("models/domain_adaptive_loss.png", caption="Domain Adaptation / Fine-tuning Loss curves", use_column_width=True)
        else:
            st.info("Fine-tuning metrics visualizer currently unavailable.")

# ==========================================================
# FOOTER
# ==========================================================
st.markdown("<br><hr style='border-top: 1px solid rgba(128, 128, 128, 0.15);'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: var(--text-color); opacity: 0.6; font-size: 13px; padding-bottom: 20px;">
    🎓 Emotion Detection & Learning Support Engine<br>
    Built with <b>TensorFlow</b> • <b>BiLSTM Architecture</b> • <b>Focal Loss</b> • <b>Streamlit</b>
</div>
""", unsafe_allow_html=True)

import streamlit as st
import cv2
import numpy as np
import base64
from io import BytesIO

# ==========================================
# 1. Configurazione e Stile (Street Night Total)
# ==========================================
st.set_page_config(page_title="Chroma Stencil Lab PRO", layout="centered")

if 'current_masks' not in st.session_state:
    st.session_state.current_masks = None
if 'current_colors' not in st.session_state:
    st.session_state.current_colors = None

def get_base64_logo(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except: return None

logo_b64 = get_base64_logo("logo.png")

# CSS REVISIONATO: Barra in primo piano e proporzioni testi fisse
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bungee&display=swap');

    /* Sfondo Street Night */
    .stApp {{
        background: radial-gradient(circle at 15% 15%, rgba(255, 230, 0, 0.25), transparent 35%),
                    radial-gradient(circle at 85% 10%, rgba(255, 230, 0, 0.2), transparent 30%),
                    linear-gradient(rgba(10, 25, 47, 0.9), rgba(10, 25, 47, 0.95)), 
                    url('https://www.transparenttextures.com/patterns/brick-wall.png');
        background-color: #0a192f;
        background-attachment: fixed;
        animation: moveBackground 70s linear infinite;
    }}

    @keyframes moveBackground {{
        from {{ background-position: 0 0; }}
        to {{ background-position: 1000px 1000px; }}
    }}

    /* --- NEWS TICKER (Sistemato per visibilità) --- */
    .ticker-wrap {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 40px;
        background-color: #000000;
        border-bottom: 2px solid #FFD700;
        z-index: 9999;
        overflow: hidden;
        display: flex;
        align-items: center;
    }}
    
    .ticker {{
        display: inline-block;
        white-space: nowrap;
        padding-left: 100%;
        animation: ticker 60s linear infinite;
        font-family: 'Bungee', cursive !important;
        color: #ffffff !important;
        font-size: 1rem;
        line-height: 40px;
    }}

    @keyframes ticker {{
        0% {{ transform: translateX(0); }}
        100% {{ transform: translateX(-200%); }}
    }}

    /* Spazio per non coprire il contenuto con la barra fissa */
    .main-content {{
        margin-top: 60px;
    }}

    .block-container {{
        max-width: 850px;
        background-color: transparent !important;
        padding-top: 50px !important;
    }}

    /* LOGO */
    .logo-img {{
        display: block;
        margin: 0 auto 20px auto;
        max-height: 220px;
        width: auto;
        filter: drop-shadow(2px 4px 6px #000);
    }}

    /* TESTI E TITOLI */
    h1, h2, h3, h4 {{
        font-family: 'Bungee', cursive !important;
        color: #FFD700 !important;
        text-shadow: 3px 3px 0px #000 !important;
        text-align: center;
        line-height: 1.2 !important;
    }}

    p, span, label, .stMarkdown {{
        color: #ffffff !important;
        font-weight: 800 !important;
        text-shadow: 1px 1px 3px #000;
    }}

    /* Selettore File (Fix visivo) */
    [data-testid="stFileUploader"] {{
        background-color: rgba(0, 0, 0, 0.5) !important;
        border: 2px dashed #FFD700 !important;
        border-radius: 15px !important;
    }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Interfaccia
# ==========================================

# 1. NEWS TICKER (Fisso in alto)
news = [
    "LERCIO: Scoperto stencil talmente realistico che il muro ha chiesto il permesso di soggiorno",
    "LERCIO: Ragazzo spruzza vernice giallo oro: scambiato per il Re Mida della Garbatella",
    "LERCIO: Studio shock: gli stencil non coprono i debiti, solo le crepe nei muri",
    "LERCIO: Nuovo spray al gusto pizza: i writer ora mangiano direttamente dai muri",
    "LERCIO: Graffiti IA generano stencil di politici onesti: l'app va in crash per mancanza di dati"
]
ticker_text = " • ".join(news)
st.markdown(f'<div class="ticker-wrap"><div class="ticker">{ticker_text}</div></div>', unsafe_allow_html=True)

# 2. LOGO
st.markdown('<div class="main-content">', unsafe_allow_html=True)
if logo_b64:
    st.markdown(f'<img src="data:image/png;base64,{logo_b64}" class="logo-img">', unsafe_allow_html=True)
else:
    st.markdown("<h1>🎨 CHROMA STENCIL LAB</h1>", unsafe_allow_html=True)

# 3. TABS
tab_ed, tab_info = st.tabs(["🏗️ EDITOR STENCIL", "⚡ INFO & TOOLS"])

with tab_ed:
    up = st.file_uploader("Carica una foto", type=["jpg", "png", "jpeg"])
    if up:
        img_raw = cv2.imdecode(np.frombuffer(up.read(), np.uint8), 1)
        st.image(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        with st.expander("⚙️ Parametri Tecnici", expanded=True):
            c1, c2 = st.columns(2)
            n_l = c1.slider("Numero Strati", 2, 8, 4)
            b_l = c1.slider("Ponti", 10, 80, 30)
            b_t = c2.slider("Spessore", 1, 10, 2)
            c_s = c2.slider("Crocette", 10, 50, 20)

        if st.button("✨ ELABORA STENCIL"):
            # (Logica di elaborazione invariata)
            img_lab = cv2.cvtColor(img_raw, cv2.COLOR_BGR2LAB)
            data = img_lab.reshape((-1, 3)).astype(np.float32)
            _, label, centers = cv2.kmeans(data, n_l, None, (cv2.TERM_CRITERIA_EPS+20, 20, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            q_lab = centers[label.flatten()].reshape((img_lab.shape))
            masks, colors = [], []
            for i in range(n_l):
                m = cv2.inRange(q_lab, centers[i], centers[i])
                masks.append(cv2.bitwise_not(m))
                rgb = cv2.cvtColor(np.uint8([[centers[i]]]), cv2.COLOR_LAB2RGB)[0][0]
                colors.append('#%02x%02x%02x' % tuple(rgb))
            st.session_state.current_masks = masks
            st.session_state.current_colors = colors
            st.success("Stencil generato con successo!")

with tab_info:
    st.markdown("## 🎨 CALCOLATORE BOMBOLETTE")
    if st.session_state.current_masks:
        for i, m in enumerate(st.session_state.current_masks):
            coverage = (np.sum(m == 0) / m.size)
            cans = max(0.2, round(coverage * 1.5, 1))
            col_a, col_b = st.columns([1, 4])
            col_a.color_picker(f"L{i+1}", st.session_state.current_colors[i], key=f"info_c_{i}", disabled=True)
            col_b.write(f"Quantità: **{cans}** bombolette (400ml)")
    else:
        st.info("Elabora un'immagine nell'Editor per vedere il preventivo spray!")

    st.markdown("---")
    st.markdown("## 🤖 IA STREET GENERATOR")
    c1, c2 = st.columns([2,1])
    c1.text_input("Testo del Graffito", "STREET ART")
    c2.selectbox("Stile", ["Wildstyle", "Bubble", "Stencil Art"])
    st.button("🚀 GENERA CON IA")

st.markdown('</div>', unsafe_allow_html=True)

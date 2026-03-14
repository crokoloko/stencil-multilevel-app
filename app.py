import streamlit as st
import cv2
import numpy as np
import base64
import zipfile
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

# CSS REVISIONATO: Centratura menu e Fix Uploader
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bungee&display=swap');

    /* Sfondo Street Night */
    .stApp {{
        background: radial-gradient(circle at 15% 15%, rgba(255, 230, 0, 0.2), transparent 35%),
                    radial-gradient(circle at 85% 10%, rgba(255, 230, 0, 0.15), transparent 30%),
                    linear-gradient(rgba(10, 25, 47, 0.92), rgba(10, 25, 47, 0.95)), 
                    url('https://www.transparenttextures.com/patterns/brick-wall.png');
        background-color: #0a192f;
        background-attachment: fixed;
        animation: moveBackground 70s linear infinite;
    }}

    @keyframes moveBackground {{
        from {{ background-position: 0 0; }}
        to {{ background-position: 1000px 1000px; }}
    }}

    /* NEWS TICKER FISSO */
    .ticker-wrap {{
        position: fixed;
        top: 0; left: 0; width: 100%; height: 40px;
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
    }}
    @keyframes ticker {{
        0% {{ transform: translateX(0); }}
        100% {{ transform: translateX(-250%); }}
    }}

    /* CENTRATURA MENU TABS */
    .stTabs [data-baseweb="tab-list"] {{
        display: flex;
        justify-content: center !important;
        width: 100%;
        gap: 20px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px 10px 0 0;
        color: #fff !important;
        padding: 10px 30px;
        font-family: 'Bungee', cursive;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: #FFD700 !important;
        color: #000 !important;
    }}

    /* FIX UPLOADER - SEMPLICE E VISIBILE */
    [data-testid="stFileUploader"] {{
        background-color: rgba(255, 255, 255, 0.05);
        border: 2px solid #FFD700 !important;
        border-radius: 15px;
        padding: 10px;
    }}
    [data-testid="stFileUploaderSection"] {{
        padding: 20px;
    }}
    /* Centratura testi */
    h1, h2, h3, h4, .stMarkdown, p, label {{
        text-align: center !important;
        font-family: 'Bungee', cursive !important;
        color: #FFD700 !important;
    }}
    p, span {{
        color: #ffffff !important;
        text-shadow: 1px 1px 2px #000;
    }}

    .logo-img {{
        display: block;
        margin: 60px auto 20px auto;
        max-height: 220px;
        width: auto;
    }}

    .stButton>button {{
        background: #FFD700 !important;
        color: black !important;
        border: 2px solid #000 !important;
        box-shadow: 4px 4px 0px #000;
        border-radius: 12px;
        width: 100%;
    }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Interfaccia
# ==========================================

# NEWS TICKER
news = [
    "LERCIO: Scoperto stencil talmente realistico che il muro ha chiesto il permesso di soggiorno",
    "LERCIO: Ragazzo spruzza vernice giallo oro: scambiato per il Re Mida della Garbatella",
    "LERCIO: Studio shock: gli stencil non coprono i debiti, solo le crepe nei muri",
    "LERCIO: Graffiti IA generano stencil di politici onesti: l'app va in crash"
]
ticker_text = " • ".join(news)
st.markdown(f'<div class="ticker-wrap"><div class="ticker">{ticker_text}</div></div>', unsafe_allow_html=True)

# LOGO
if logo_b64:
    st.markdown(f'<img src="data:image/png;base64,{logo_b64}" class="logo-img">', unsafe_allow_html=True)
else:
    st.markdown("<h1 style='margin-top:70px;'>🎨 CHROMA STENCIL LAB</h1>", unsafe_allow_html=True)

# MENU TABS CENTRATO
tab_ed, tab_info = st.tabs(["🏗️ EDITOR", "⚡ INFO & TOOLS"])

with tab_ed:
    st.markdown("### CARICA LA TUA FOTO")
    up = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    
    if up:
        img_raw = cv2.imdecode(np.frombuffer(up.read(), np.uint8), 1)
        st.image(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        with st.expander("⚙️ SETTAGGI TECNICI", expanded=True):
            n_l = st.slider("Livelli Colore", 2, 8, 4)
            b_l = st.slider("Ponti", 10, 80, 30)
            c_s = st.slider("Crocette", 10, 50, 20)

        if st.button("✨ ELABORA STENCIL"):
            # Logica K-means
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
            
            res_p, res_l = st.tabs(["🌌 ANTEPRIMA", "✂️ TAGLIO"])
            with res_p:
                st.success("Stencil generato!")
                # Qui andrebbe la funzione preview...
            with res_l:
                l_tabs = st.tabs([f"{i+1}" for i in range(n_l)])
                for i, lt in enumerate(l_tabs):
                    with lt:
                        st.image(masks[i], use_container_width=True)
                        _, buf = cv2.imencode(".png", masks[i])
                        st.download_button(f"Scarica PNG {i+1}", buf.tobytes(), f"L{i+1}.png", key=f"dl_{i}")

with tab_info:
    st.markdown("## 🎨 CAN BUDGET")
    if st.session_state.current_masks:
        for i, m in enumerate(st.session_state.current_masks):
            coverage = (np.sum(m == 0) / m.size)
            cans = max(0.2, round(coverage * 1.5, 1))
            st.write(f"Strato {i+1}: **{cans}** bombolette")
    else:
        st.write("Elabora un'immagine per il calcolo.")
    
    st.markdown("---")
    st.markdown("## 🤖 IA GENERATOR")
    st.text_input("Soggetto Graffito", "STREET ART")
    st.button("🚀 GENERA")

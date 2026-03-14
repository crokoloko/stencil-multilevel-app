import streamlit as st
import cv2
import numpy as np
import zipfile
import base64
from io import BytesIO

# ==========================================
# 1. Configurazione e Stile (Urban Wall Animato)
# ==========================================
st.set_page_config(page_title="Chroma Stencil Lab PRO", layout="centered")

if 'saved_projects' not in st.session_state:
    st.session_state.saved_projects = []

# Funzione per caricare il logo
def get_base64_logo(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return None

logo_b64 = get_base64_logo("logo.png")

# CSS con Doppie Graffe per evitare errori Python e Sfondo Animato
st.markdown(f"""
<style>
    /* Sfondo principale: Muro di Mattoni Notturno Animato */
    .stApp {{
        background: radial-gradient(circle at 15% 15%, rgba(255, 230, 0, 0.25), transparent 35%),
                    radial-gradient(circle at 85% 10%, rgba(255, 230, 0, 0.2), transparent 30%),
                    linear-gradient(rgba(10, 25, 47, 0.9), rgba(10, 25, 47, 0.95)), 
                    url('https://www.transparenttextures.com/patterns/brick-wall.png');
        background-color: #0a192f;
        background-attachment: fixed;
        background-blend-mode: overlay, overlay, normal, normal;
        /* ANIMAZIONE MOVIMENTO MATTONI */
        animation: moveBackground 60s linear infinite, lampFlicker 8s ease-in-out infinite alternate;
    }}

    @keyframes moveBackground {{
        from {{ background-position: 0 0, 0 0, 0 0, 0 0; }}
        to {{ background-position: 1000px 1000px, 200px 400px, 0 0, 0 0; }}
    }}

    @keyframes lampFlicker {{
        0% {{ filter: brightness(1); }}
        50% {{ filter: brightness(1.1); }}
        100% {{ filter: brightness(0.95); }}
    }}

    .stApp::before {{
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        backdrop-filter: blur(5px);
        z-index: -1;
    }}

    .block-container {{
        max-width: 850px;
        background-color: transparent !important;
        box-shadow: none !important;
        padding: 20px !important;
        margin: auto;
    }}

    /* Testi e Titoli Bianco su Scuro */
    h1, h2, h3, h4, p, span {{
        color: #ffffff !important;
        font-weight: 800 !important;
        text-align: center;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.9);
    }}

    .stSlider label, .stFileUploader label {{
        color: #FFD700 !important;
        font-weight: bold !important;
        text-shadow: 1px 1px 3px #000;
    }}

    .logo-img {{
        display: block;
        margin: 0 auto 30px auto;
        max-height: 250px;
        filter: drop-shadow(2px 5px 10px rgba(0,0,0,0.7));
    }}

    .stTabs [data-baseweb="tab-list"] {{
        justify-content: center;
        gap: 15px;
    }}

    .stTabs [data-baseweb="tab"] {{
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px 10px 0 0;
        color: #fff !important;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: #FFD700 !important;
        color: #000 !important;
    }}

    .stButton>button {{
        background: #FFD700 !important;
        color: black !important;
        border: 2px solid #000 !important;
        box-shadow: 4px 4px 0px #000;
        border-radius: 12px;
        font-weight: bold;
    }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Funzioni Core
# ==========================================

def apply_stencil_logic(mask, b_len, b_thick, cross_size):
    h, w = mask.shape
    out = mask.copy()
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, cnt in enumerate(contours):
            if hierarchy[i][3] != -1 and cv2.contourArea(cnt) > 80:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                    cv2.line(out, (cX, cY), (cX, cY - b_len), 255, b_thick)
    m = cross_size + 25
    centers = [(m, m), (w-m, m), (m, h-m), (w-m, h-m)]
    for cX, cY in centers:
        cv2.line(out, (cX - cross_size, cY), (cX + cross_size, cY), 255, b_thick)
        cv2.line(out, (cX, cY - cross_size), (cX, cY + cross_size), 255, b_thick)
    return out

def create_preview(masks, colors):
    h, w = masks[0].shape
    bg = np.full((h, w, 3), 60, dtype=np.uint8)
    noise = np.random.normal(0, 15, (h, w, 3)).astype(np.int16)
    canvas = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for i, m in enumerate(masks):
        rgb = tuple(int(colors[i].lstrip('#')[j:j+2], 16) for j in (0, 2, 4))
        bgr = (rgb[2], rgb[1], rgb[0])
        color_img = np.full((h, w, 3), bgr, dtype=np.uint8)
        layer_c = cv2.bitwise_and(color_img, color_img, mask=m)
        bg_part = cv2.bitwise_and(canvas, canvas, mask=m)
        blended = cv2.addWeighted(bg_part, 0.4, layer_c, 0.6, 0)
        canvas = cv2.bitwise_and(canvas, canvas, mask=cv2.bitwise_not(m))
        canvas = cv2.add(canvas, blended)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

def generate_zip(project):
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        prev_bgr = cv2.cvtColor(project['preview'], cv2.COLOR_RGB2BGR)
        _, p_img = cv2.imencode(".png", prev_bgr)
        z.writestr("anteprima.png", p_img.tobytes())
        for i, m in enumerate(project['masks']):
            _, m_img = cv2.imencode(".png", m)
            z.writestr(f"strato_{i+1}.png", m_img.tobytes())
    return buf.getvalue()

# ==========================================
# 3. Interfaccia
# ==========================================

if logo_b64:
    st.markdown(f'<img src="data:image/png;base64,{logo_b64}" class="logo-img">', unsafe_allow_html=True)
else:
    st.title("🌈 CHROMA STENCIL LAB")

t_editor, t_saved = st.tabs(["🏗️ EDITOR", "💾 SALVATI"])

with t_editor:
    up = st.file_uploader("Carica foto", type=["jpg", "png", "jpeg"])
    if up:
        img_raw = cv2.imdecode(np.frombuffer(up.read(), np.uint8), 1)
        st.image(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        with st.expander("⚙️ Parametri Tecnici", expanded=True):
            c1, c2 = st.columns(2)
            n_l = c1.slider("Livelli Colore", 2, 8, 4)
            b_l = c1.slider("Lungezza Ponti", 10, 80, 30)
            b_t = c2.slider("Spessore", 1, 10, 2)
            c_s = c2.slider("Crocette", 10, 50, 20)

        if st.button("✨ ELABORA STENCIL"):
            img_lab = cv2.cvtColor(img_raw, cv2.COLOR_BGR2LAB)
            data = img_lab.reshape((-1, 3)).astype(np.float32)
            _, label, centers = cv2.kmeans(data, n_l, None, (cv2.TERM_CRITERIA_EPS+20, 20, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            q_lab = centers[label.flatten()].reshape((img_lab.shape))

            masks, colors = [], []
            for i in range(n_l):
                m = cv2.inRange(q_lab, centers[i], centers[i])
                masks.append(apply_stencil_logic(m, b_l, b_t, c_s))
                rgb = cv2.cvtColor(np.uint8([[centers[i]]]), cv2.COLOR_LAB2RGB)[0][0]
                colors.append('#%02x%02x%02x' % tuple(rgb))
            
            res_p, res_l = st.tabs(["🌌 ANTEPRIMA", "✂️ TAGLIO"])
            with res_p:
                p_img = create_preview(masks, colors)
                st.image(p_img, use_container_width=True)
                if st.button("💾 SALVA NELL'ARCHIVIO"):
                    st.session_state.saved_projects.append({"name": f"Stencil_{len(st.session_state.saved_projects)+1}", "preview": p_img, "masks": masks, "colors": colors})
                    st.success("Salvato!")
            with res_l:
                l_tabs = st.tabs([f"{i+1}" for i in range(n_l)])
                for i, lt in enumerate(l_tabs):
                    with lt:
                        st.image(masks[i], use_container_width=True)
                        _, buf = cv2.imencode(".png", masks[i])
                        st.download_button(f"Scarica {i+1}", buf.tobytes(), f"L{i+1}.png", key=f"dl_{i}")

with t_saved:
    if not st.session_state.saved_projects:
        st.info("Nessun progetto salvato.")
    else:
        for idx, p in enumerate(st.session_state.saved_projects):
            with st.expander(f"📁 {p['name']}"):
                st.image(p['preview'], use_container_width=True)
                z_data = generate_zip(p)
                st.download_button("📥 SCARICA ZIP", z_data, f"{p['name']}.zip", key=f"z_{idx}")
                if st.button(f"🗑️ Elimina", key=f"del_{idx}"):
                    st.session_state.saved_projects.pop(idx); st.rerun()

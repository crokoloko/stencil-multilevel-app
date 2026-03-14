import streamlit as st
import cv2
import numpy as np
import zipfile
import base64
from io import BytesIO

# ==========================================
# 1. Configurazione e Stile (Street Night Total)
# ==========================================
st.set_page_config(page_title="Chroma Stencil Lab PRO", layout="centered")

# Inizializzazione variabili di sessione
if 'saved_projects' not in st.session_state:
    st.session_state.saved_projects = []
if 'current_masks' not in st.session_state:
    st.session_state.current_masks = None

def get_base64_logo(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except: return None

logo_b64 = get_base64_logo("logo.png")

# CSS Revisionato (Doppie graffe {{ }} per evitare errori Python)
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bungee&display=swap');

    .stApp {{
        background: radial-gradient(circle at 15% 15%, rgba(255, 230, 0, 0.25), transparent 35%),
                    radial-gradient(circle at 85% 10%, rgba(255, 230, 0, 0.2), transparent 30%),
                    linear-gradient(rgba(10, 25, 47, 0.9), rgba(10, 25, 47, 0.95)), 
                    url('https://www.transparenttextures.com/patterns/brick-wall.png');
        background-color: #0a192f;
        background-attachment: fixed;
        animation: moveBackground 70s linear infinite, lampFlicker 10s ease-in-out infinite alternate;
    }}

    @keyframes moveBackground {{
        from {{ background-position: 0 0; }}
        to {{ background-position: 1000px 1000px; }}
    }}

    @keyframes lampFlicker {{
        0% {{ filter: brightness(1); }}
        50% {{ filter: brightness(1.15); }}
        100% {{ filter: brightness(0.9); }}
    }}

    .block-container {{
        max-width: 850px;
        background-color: transparent !important;
        padding: 20px !important;
    }}

    /* Font Street */
    h1, h2, h3, h4 {{
        font-family: 'Bungee', cursive !important;
        color: #FFD700 !important;
        text-align: center;
        text-shadow: 3px 3px 0px #000 !important;
    }}

    p, span, label, .stMarkdown {{
        color: #ffffff !important;
        font-weight: 800 !important;
        text-shadow: 2px 2px 4px #000;
    }}

    /* Fix Uploader */
    [data-testid="stFileUploader"] {{
        background-color: rgba(0, 0, 0, 0.5) !important;
        border: 2px dashed #FFD700 !important;
        border-radius: 15px !important;
        padding: 20px !important;
    }}

    .logo-img {{
        display: block;
        margin: 0 auto 30px auto;
        max-height: 250px;
        filter: drop-shadow(2px 4px 6px #000);
    }}

    .stTabs [data-baseweb="tab-list"] {{ justify-content: center; gap: 15px; }}
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
        font-family: 'Bungee', sans-serif;
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
        layer_c = cv2.bitwise_and(color_img, color_img, mask=cv2.bitwise_not(m))
        bg_part = cv2.bitwise_and(canvas, canvas, mask=cv2.bitwise_not(m))
        blended = cv2.addWeighted(bg_part, 0.4, layer_c, 0.6, 0)
        canvas = cv2.bitwise_and(canvas, canvas, mask=m)
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
            z.writestr(f"strato_{i+1}_{project['colors'][i].replace('#','')}.png", m_img.tobytes())
    return buf.getvalue()

# ==========================================
# 3. Interfaccia
# ==========================================

if logo_b64:
    st.markdown(f'<img src="data:image/png;base64,{logo_b64}" class="logo-img">', unsafe_allow_html=True)
else:
    st.title("🌈 CHROMA STENCIL LAB")

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
            with st.spinner('Calcolo in corso...'):
                img_lab = cv2.cvtColor(img_raw, cv2.COLOR_BGR2LAB)
                data = img_lab.reshape((-1, 3)).astype(np.float32)
                _, label, centers = cv2.kmeans(data, n_l, None, (cv2.TERM_CRITERIA_EPS+20, 20, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
                centers = np.uint8(centers)
                q_lab = centers[label.flatten()].reshape((img_lab.shape))

                masks, colors = [], []
                for i in range(n_l):
                    m = cv2.inRange(q_lab, centers[i], centers[i])
                    masks.append(apply_stencil_logic(cv2.bitwise_not(m), b_l, b_t, c_s))
                    rgb = cv2.cvtColor(np.uint8([[centers[i]]]), cv2.COLOR_LAB2RGB)[0][0]
                    colors.append('#%02x%02x%02x' % tuple(rgb))
                
                st.session_state.current_masks = masks
                st.session_state.current_colors = colors
                
                t_prev, t_lay = st.tabs(["🌌 ANTEPRIMA", "✂️ TAGLIO"])
                with t_prev:
                    p_img = create_preview(masks, colors)
                    st.image(p_img, use_container_width=True)
                    if st.button("💾 SALVA NEI PREFERITI"):
                        st.session_state.saved_projects.append({"name": f"Project_{len(st.session_state.saved_projects)+1}", "preview": p_img, "masks": masks, "colors": colors})
                        st.success("Salvato!")
                with t_lay:
                    l_tabs = st.tabs([f"{i+1}" for i in range(n_l)])
                    for i, lt in enumerate(l_tabs):
                        with lt:
                            st.image(masks[i], use_container_width=True)
                            _, buf = cv2.imencode(".png", masks[i])
                            st.download_button(f"Download {i+1}", buf.tobytes(), f"L{i+1}.png", key=f"dl_{i}")

with tab_info:
    # --- IA GENERATOR ---
    st.markdown("## 🤖 IA STREET GENERATOR")
    with st.container():
        st.write("Genera graffiti complessi con l'IA (Simulazione)")
        c1, c2 = st.columns([2,1])
        prompt = c1.text_input("Testo del Graffito", "STREET ART")
        style = c2.selectbox("Stile", ["Wildstyle", "Bubble", "Stencil Art"])
        if st.button("🚀 GENERA CON IA"):
            st.info(f"Sto creando un pezzo in stile {style} per '{prompt}'...")
            st.warning("⚠️ Per attivare la generazione reale, collega le API di DALL-E o Stable Diffusion.")

    st.markdown("---")

    # --- CALCOLATORE BOMBOLETTE ---
    st.markdown("## 🎨 CALCOLATORE BOMBOLETTE")
    if st.session_state.current_masks:
        for i, m in enumerate(st.session_state.current_masks):
            coverage = (np.sum(m == 0) / m.size)
            cans = max(0.2, round(coverage * 1.5, 1))
            col_a, col_b = st.columns([1, 4])
            col_a.color_picker(f"L{i+1}", st.session_state.current_colors[i], key=f"info_c_{i}", disabled=True)
            col_b.write(f"Servono circa **{cans}** bombolette (400ml)")
    else:
        st.write("Carica ed elabora una foto nell'Editor per vedere il preventivo spray!")

    st.markdown("---")

    # --- ARCHIVIO (EX SALVATI) ---
    st.markdown("## 💾 I TUOI SALVATI")
    if not st.session_state.saved_projects:
        st.write("Nessun progetto in archivio.")
    else:
        for idx, p in enumerate(st.session_state.saved_projects):
            with st.expander(f"📁 {p['name']}"):
                st.image(p['preview'], use_container_width=True)
                z_data = generate_zip(p)
                st.download_button("📥 SCARICA ZIP", z_data, f"{p['name']}.zip", key=f"zip_{idx}")
                if st.button("🗑️ Elimina", key=f"del_{idx}"):
                    st.session_state.saved_projects.pop(idx); st.rerun()


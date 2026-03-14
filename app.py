import streamlit as st
import cv2
import numpy as np
import zipfile
import base64
from io import BytesIO

# ==========================================
# 1. Configurazione e Stile (Total Urban Wall)
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

# CSS Aggiornato per rimuovere il riquadro e adattare il contrasto
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
        animation: moveBackground 60s linear infinite, lampFlicker 8s ease-in-out infinite alternate;
    }}

    @keyframes moveBackground {{
        from {{ background-position: 0 0, 0 0, 0 0, 0 0; }}
        to {{ background-position: 500px 1000px, 200px 400px, 0 0, 0 0; }}
    }}

    @keyframes lampFlicker {{
        0% {{ filter: brightness(1) contrast(1); }}
        50% {{ filter: brightness(1.1) contrast(1.05); }}
        100% {{ filter: brightness(0.95) contrast(0.98); }}
    }}

    /* Sfocatura dello sfondo tramite un overlay fisso */
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        backdrop-filter: blur(5px);
        z-index: -1;
    }}

    /* MODIFICA CRUCCIALE: Rimozione del riquadro bianco panna */
    .block-container {{
        max-width: 850px;
        background-color: transparent !important; /* Totalmente trasparente */
        box-shadow: none !important; /* Rimuove l'ombra del riquadro */
        padding: 20px !important;
        margin: auto;
    }}

    /* ADATTAMENTO CONTRASTO TESTI (Sfondo Scuro) */
    h1, h2, h3, h4, p {{
        color: #ffffff !important; /* Testo bianco */
        font-weight: 800 !important;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
    }}

    /* Etichette degli slider e widget (Giallo acceso) */
    .stSlider label, .stFileUploader label, .stSelectbox label {{
        color: #FFD700 !important;
        font-weight: bold;
    }}

    h1 {{ 
        font-family: 'Bungee', cursive; 
        color: #FFD700 !important; 
        text-shadow: 3px 3px 0px #000;
        margin-bottom: 30px;
    }}

    /* Box Info Report (Semi-trasparente scuro) */
    .spray-info-box {{
        background-color: rgba(0, 0, 0, 0.5);
        border: 2px solid #FFD700;
        border-left: 10px solid #FFD700;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 30px;
        color: #fff;
    }

    /* Tabs e Menu (Adattati allo sfondo scuro) */
    .stTabs [data-baseweb="tab-list"] {{
        justify-content: center;
        gap: 15px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px 10px 0 0;
        color: #fff !important;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {{
        background-color: #FFD700 !important;
        border: 2px solid #000 !important;
        color: #000 !important;
    }}

    /* Pulsante Giallo Dorato */
    .stButton>button {{
        background: #FFD700 !important;
        color: black !important;
        border: 2px solid #000 !important;
        box-shadow: 4px 4px 0px #000;
        border-radius: 12px;
        height: 55px;
        font-size: 1.1rem;
    }}
    .stButton>button:hover {{
        background: #FFEA00 !important;
    }}

    /* Logo XL centrato */
    .logo-img {{
        display: block;
        margin-left: auto;
        margin-right: auto;
        max-height: 250px;
        width: auto;
        margin-bottom: 30px;
        filter: drop-shadow(2px 4px 6px rgba(0,0,0,0.8));
    }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Funzioni Core (Invariate)
# ==========================================

def apply_bridges_and_crosses(mask, b_len, b_thick, cross_size):
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
    m = cross_size + 20
    centers = [(m, m), (w-m, m), (m, h-m), (w-m, h-m)]
    for cX, cY in centers:
        cv2.line(out, (cX - cross_size, cY), (cX + cross_size, cY), 255, b_thick)
        cv2.line(out, (cX, cY - cross_size), (cX, cY + cross_size), 255, b_thick)
    return out

def create_preview(masks, colors):
    h, w = masks[0].shape
    bg = np.full((h, w, 3), 100, dtype=np.uint8)
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
        preview_bgr = cv2.cvtColor(project['preview'], cv2.COLOR_RGB2BGR)
        _, p_img = cv2.imencode(".png", preview_bgr)
        z.writestr("anteprima.png", p_img.tobytes())
        for i, m in enumerate(project['masks']):
            _, m_img = cv2.imencode(".png", m)
            z.writestr(f"strato_{i+1}_{project['colors'][i]}.png", m_img.tobytes())
    return buf.getvalue()

# ==========================================
# 3. Interfaccia
# ==========================================

# Logo Centrale e Grande
if logo_b64:
    st.markdown(f'<img src="data:image/png;base64,{logo_b64}" class="logo-img">', unsafe_allow_html=True)
else:
    st.title("🌈 CHROMA STENCIL LAB")

# Menu centrato
tab_ed, tab_sav = st.tabs(["🏗️ EDITOR PROGETTO", "💾 SALVATI"])

with tab_ed:
    up = st.file_uploader("1. Carica la foto da trasformare", type=["jpg", "png", "jpeg"])
    if up:
        img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), 1)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Originale")
        
        with st.expander("⚙️ Parametri Tecnici", expanded=True):
            c1, c2 = st.columns(2)
            n_l = c1.slider("Numero strati", 2, 8, 4)
            b_l = c1.slider("Lunghezza ponti", 10, 80, 30)
            b_t = c2.slider("Spessore linee", 1, 10, 2)
            c_s = c2.slider("Taglia crocette", 10, 50, 20)

        if st.button("✨ ELABORA STENCIL"):
            with st.spinner('Creazione stencil in corso...'):
                # Processing K-Means
                img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                data = img_lab.reshape((-1, 3)).astype(np.float32)
                _, label, centers = cv2.kmeans(data, n_l, None, (cv2.TERM_CRITERIA_EPS+20, 20, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
                centers = np.uint8(centers)
                quantized_lab = centers[label.flatten()].reshape((img_lab.shape))

                masks, colors = [], []
                for i in range(n_l):
                    m = cv2.inRange(quantized_lab, centers[i], centers[i])
                    masks.append(apply_bridges_and_crosses(m, b_l, b_t, c_s))
                    rgb = cv2.cvtColor(np.uint8([[centers[i]]]), cv2.COLOR_LAB2RGB)[0][0]
                    colors.append('#%02x%02x%02x' % tuple(rgb))
                
                # Menu Risultati
                t_prev, t_lay = st.tabs(["🌌 ANTEPRIMA", "✂️ TAGLIO"])
                with t_prev:
                    p_img = create_preview(masks, colors)
                    st.image(p_img, use_container_width=True, caption="Simulazione finale")
                    
                    st.markdown("### 🖌️ Colori Rilevati")
                    col_c = st.columns(n_l)
                    for i in range(n_l):
                        col_c[i].color_picker(f"L{i+1}", colors[i], key=f"cp_ed_{i}")

                    if st.button("💾 SALVA PROGETTO NELL'ARCHIVIO"):
                        st.session_state.saved_projects.append({
                            "name": f"Stencil_{len(st.session_state.saved_projects)+1}", 
                            "preview": p_img, 
                            "masks": masks, 
                            "colors": colors
                        })
                        st.success("Aggiunto ai Salvati!")

                with t_lay:
                    st.info("Ritaglia le parti NERE. Le crocette servono per l'allineamento.")
                    l_tabs = st.tabs([f"{i+1}" for i in range(n_l)])
                    for i, lt in enumerate(l_tabs):
                        with lt:
                            st.image(masks[i], use_container_width=True, caption=f"Maschera {i+1}")
                            _, buf = cv2.imencode(".png", masks[i])
                            st.download_button(f"📥 Scarica PNG {i+1}", buf.tobytes(), f"L{i+1}.png", key=f"d_{i}")

with tab_saved:
    if not st.session_state.saved_projects:
        st.info("Non ci sono ancora progetti salvati.")
    else:
        for idx, p in enumerate(st.session_state.saved_projects):
            with st.expander(f"📁 {p['name']} - {len(p['masks'])} strati"):
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    st.image(p['preview'], use_container_width=True)
                with col_b:
                    st.write("📦 **Pacchetto Progetto**")
                    zip_data = generate_zip(p)
                    st.download_button(
                        label="📥 SCARICA ZIP COMPLETO",
                        data=zip_data,
                        file_name=f"{p['name']}.zip",
                        mime="application/zip",
                        key=f"zip_{idx}"
                    )
                    st.markdown("---")
                    if st.button(f"🗑️ Elimina {p['name']}", key=f"del_{idx}"):
                        st.session_state.saved_projects.pop(idx)
                        st.rerun()

import streamlit as st
import cv2
import numpy as np
import zipfile
import base64
from io import BytesIO
from PIL import Image

# ==========================================
# 1. Configurazione e Stile (Logo e Centratura)
# ==========================================
st.set_page_config(page_title="Chroma Stencil Lab PRO", layout="centered")

if 'saved_projects' not in st.session_state:
    st.session_state.saved_projects = []

# --- FUNZIONE PER CARICARE IL LOGO LOCALE ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- CSS CON ANIMAZIONE, LOGO E CENTRATURA ---
# Assicurati che 'logo.png' sia nella stessa cartella di app.py
try:
    logo_base64 = get_base64_of_bin_file('logo.png')
    logo_html = f'<img src="data:image/png;base64,{logo_base64}" class="app-logo">'
except FileNotFoundError:
    logo_html = '<h1 style="color: #DAA520;">🎨</h1>' # Fallback se il logo non c'è

st.markdown(f"""
<style>
    /* Sfondo animato: Muro di mattoni sfocato */
    .stApp {{
        background: linear-gradient(rgba(0,0,0,0.4), rgba(0,0,0,0.4)), 
                    url('https://www.transparenttextures.com/patterns/brick-wall.png');
        background-color: #FFFDD0;
        background-attachment: fixed;
        animation: moveBackground 40s linear infinite;
    }}

    @keyframes moveBackground {{
        from {{ background-position: 0 0; }}
        to {{ background-position: 500px 1000px; }}
    }}

    /* Sfocatura dello sfondo tramite un overlay */
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        backdrop-filter: blur(4px);
        z-index: -1;
    }}

    /* Centratura del contenuto principale */
    .block-container {{
        max-width: 800px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        margin: auto;
        background-color: rgba(255, 253, 208, 0.6); /* Sfondo panna semi-trasparente per il contenuto */
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }}

    /* Stile per il Logo personalizzato */
    .logo-container {{
        text-align: center;
        margin-bottom: 20px;
    }}
    .app-logo {{
        max-height: 100px; /* Regola l'altezza del logo */
        width: auto;
    }}

    /* ALTO CONTRASTO PER LE SCRITTE */
    h1, h2, h3, h4, p, label, .stMarkdown {{
        color: #000000 !important;
        font-weight: 800 !important;
    }}

    h1 {{ 
        font-family: 'Bungee', cursive; 
        color: #DAA520 !important; 
        text-shadow: 2px 2px 0px #000;
        text-align: center;
    }}

    /* Box Info Report (Semi-trasparente per vedere il muro) */
    .spray-info-box {{
        background-color: rgba(245, 245, 220, 0.9);
        border: 2px solid #000;
        border-left: 10px solid #FFD700;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 25px;
        color: #000;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: rgba(0,0,0,0.1);
        border-radius: 8px 8px 0 0;
        color: #000 !important;
        font-weight: bold;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: #FFD700 !important;
        border: 2px solid #000 !important;
    }}

    /* Pulsante Genera */
    .stButton>button {{
        background: #FFD700 !important;
        color: black !important;
        border: 2px solid #000 !important;
        box-shadow: 4px 4px 0px #000;
    }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Funzioni Core (Logica Invariata)
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
        bg_p = cv2.bitwise_and(canvas, canvas, mask=m)
        blended = cv2.addWeighted(bg_p, 0.4, layer_c, 0.6, 0)
        canvas = cv2.bitwise_and(canvas, canvas, mask=cv2.bitwise_not(m))
        canvas = cv2.add(canvas, blended)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

def generate_zip(project):
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        preview_bgr = cv2.cvtColor(project['preview'], cv2.COLOR_RGB2BGR)
        _, prev_img = cv2.imencode(".png", preview_bgr)
        z.writestr("00_anteprima.png", prev_img.tobytes())
        color_summary = "COLORI:\n"
        for i, mask in enumerate(project['masks']):
            _, m_img = cv2.imencode(".png", mask)
            z.writestr(f"strato_{i+1}_{project['colors'][i]}.png", m_img.tobytes())
            color_summary += f"Strato {i+1}: {project['colors'][i]}\n"
        z.writestr("info.txt", color_summary)
    return buf.getvalue()

# ==========================================
# 3. Flusso Applicazione
# ==========================================

# Visualizzazione del Logo centrato
st.markdown(f'<div class="logo-container">{logo_html}</div>', unsafe_allow_html=True)
st.title("🌈 Chroma Stencil Lab")

tab_editor, tab_saved = st.tabs(["🏗️ EDITOR PROGETTO", "💾 ARCHIVIO SALVATI"])

with tab_editor:
    up_file = st.file_uploader("Carica una foto", type=["jpg", "png", "jpeg"])
    
    if up_file:
        img = cv2.imdecode(np.frombuffer(up_file.read(), np.uint8), 1)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Originale")
        
        with st.expander("⚙️ Parametri Tecnici", expanded=True):
            c1, c2 = st.columns(2)
            n_layers = c1.slider("Numero strati", 2, 8, 4)
            b_len = c1.slider("Ponti", 10, 80, 30)
            b_thick = c2.slider("Spessore", 1, 10, 2)
            cross_size = c2.slider("Crocette", 10, 50, 20)

        if st.button("✨ ELABORA STENCIL"):
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            data = img_lab.reshape((-1, 3)).astype(np.float32)
            _, label, centers = cv2.kmeans(data, n_layers, None, (cv2.TERM_CRITERIA_EPS+20, 20, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            quantized_lab = centers[label.flatten()].reshape((img_lab.shape))

            masks, hex_colors = [], []
            for i in range(n_layers):
                m = cv2.inRange(quantized_lab, centers[i], centers[i])
                masks.append(apply_bridges_and_crosses(m, b_len, b_thick, cross_size))
                rgb = cv2.cvtColor(np.uint8([[centers[i]]]), cv2.COLOR_LAB2RGB)[0][0]
                hex_colors.append('#%02x%02x%02x' % tuple(rgb))
            
            st.header("🏁 Risultati")
            res_tab_prev, res_tab_layers = st.tabs(["🌌 ANTEPRIMA URBAN", "✂️ STRATI DA RITAGLIARE"])
            
            with res_tab_prev:
                preview_img = create_preview(masks, hex_colors)
                st.image(preview_img, use_container_width=True)
                
                st.markdown("### 🖌️ Tinte")
                col_c = st.columns(n_layers)
                for i in range(n_layers):
                    col_c[i].color_picker(f"{i+1}", hex_colors[i], key=f"cp_ed_{i}")

                if st.button("💾 SALVA PROGETTO"):
                    st.session_state.saved_projects.append({
                        "name": f"Progetto_{len(st.session_state.saved_projects)+1}",
                        "preview": preview_img,
                        "masks": masks,
                        "colors": hex_colors
                    })
                    st.success("Salvato!")

            with res_tab_layers:
                layer_tabs = st.tabs([f"{i+1}" for i in range(n_layers)])
                for i, l_tab in enumerate(layer_tabs):
                    with l_tab:
                        col_m, col_i = st.columns([2, 1])
                        col_m.image(masks[i], use_container_width=True)
                        with col_i:
                            st.write(f"**Livello {i+1}**")
                            _, buf = cv2.imencode(".png", masks[i])
                            st.download_button(f"📥 Scarica {i+1}", buf.tobytes(), f"strato_{i+1}.png", key=f"btn_l_{i}")

with tab_saved:
    if not st.session_state.saved_projects:
        st.info("Archivio vuoto.")
    else:
        for idx, proj in enumerate(st.session_state.saved_projects):
            with st.expander(f"📁 {proj['name']}"):
                col_a, col_b = st.columns([1, 1])
                col_a.image(proj['preview'], use_container_width=True)
                with col_b:
                    zip_data = generate_zip(proj)
                    st.download_button("📥 SCARICA ZIP", zip_data, f"{proj['name']}.zip", key=f"z_{idx}")
                    if st.button("🗑️ Elimina", key=f"del_{idx}"):
                        st.session_state.saved_projects.pop(idx); st.rerun()

import streamlit as st
import cv2
import numpy as np
from io import BytesIO

# ==========================================
# 1. Configurazione e Stile
# ==========================================
st.set_page_config(page_title="Chroma Stencil Lab PRO", layout="centered")

# Inizializzazione memoria salvataggi
if 'saved_projects' not in st.session_state:
    st.session_state.saved_projects = []

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    h1 { font-family: 'Bungee', cursive; color: #FFD700 !important; text-align: center; }
    .spray-info-box { background-color: #161b22; border-left: 8px solid #FFD700; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
    .stButton>button { width: 100%; border-radius: 10px; font-weight: bold; }
    .save-btn>div>button { background-color: #238636 !important; color: white !important; border: none !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Funzioni Core
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

# ==========================================
# 3. Struttura Menu principale
# ==========================================

st.title("🌈 Chroma Stencil Lab")

# Navigazione principale
main_tab_editor, main_tab_saved = st.tabs(["🏗️ EDITOR PROGETTO", "💾 SALVATI"])

with main_tab_editor:
    up_file = st.file_uploader("Carica immagine", type=["jpg", "png", "jpeg"])
    
    if up_file:
        file_bytes = np.frombuffer(up_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        with st.expander("⚙️ Parametri Tecnici", expanded=False):
            c1, c2 = st.columns(2)
            n_layers = c1.slider("Strati", 2, 8, 4)
            b_len = c1.slider("Ponti", 10, 80, 30)
            b_thick = c2.slider("Spessore", 1, 10, 2)
            cross_size = c2.slider("Crocette", 10, 50, 20)

        if st.button("✨ ELABORA STENCIL"):
            # Processing
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            data = img_lab.reshape((-1, 3)).astype(np.float32)
            _, label, centers = cv2.kmeans(data, n_layers, None, (cv2.TERM_CRITERIA_EPS+20, 20, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            quantized_lab = centers[label.flatten()].reshape((img_lab.shape))

            masks = []
            hex_colors = []
            for i in range(n_layers):
                m = cv2.inRange(quantized_lab, centers[i], centers[i])
                masks.append(apply_bridges_and_crosses(m, b_len, b_thick, cross_size))
                rgb = cv2.cvtColor(np.uint8([[centers[i]]]), cv2.COLOR_LAB2RGB)[0][0]
                hex_colors.append('#%02x%02x%02x' % tuple(rgb))
            
            # Sub-menu Risultati
            tab_prev, tab_cuts = st.tabs(["🌌 ANTEPRIMA URBAN", "✂️ STRATI DA RITAGLIARE"])
            
            with tab_prev:
                preview_img = create_preview(masks, hex_colors)
                st.image(preview_img, use_container_width=True)
                
                # FUNZIONE SALVA
                st.markdown('<div class="save-btn">', unsafe_allow_html=True)
                if st.button("💾 SALVA PROGETTO NELL'ARCHIVIO"):
                    project = {
                        "name": f"Progetto {len(st.session_state.saved_projects)+1}",
                        "preview": preview_img,
                        "masks": masks,
                        "colors": hex_colors
                    }
                    st.session_state.saved_projects.append(project)
                    st.success("Progetto salvato nella sezione 'SALVATI'!")
                st.markdown('</div>', unsafe_allow_html=True)

            with tab_cuts:
                for i in range(n_layers):
                    col_v, col_d = st.columns([2, 1])
                    col_v.image(masks[i], caption=f"Strato {i+1}")
                    _, buf = cv2.imencode(".png", masks[i])
                    col_d.download_button(f"Scarica L{i+1}", buf.tobytes(), f"layer_{i+1}.png", key=f"d_ed_{i}")

# ==========================================
# 4. Sezione SALVATI
# ==========================================
with main_tab_saved:
    if not st.session_state.saved_projects:
        st.info("Nessun progetto salvato. Crea uno stencil e clicca su 'Salva'!")
    else:
        for idx, proj in enumerate(st.session_state.saved_projects):
            with st.expander(f"📁 {proj['name']} - {len(proj['masks'])} strati"):
                col_p, col_m = st.columns([1, 1])
                with col_p:
                    st.subheader("Anteprima")
                    st.image(proj['preview'], use_container_width=True)
                with col_m:
                    st.subheader("Download Strati")
                    for i, m in enumerate(proj['masks']):
                        _, b = cv2.imencode(".png", m)
                        st.download_button(f"Download Strato {i+1} ({proj['colors'][i]})", b.tobytes(), f"{proj['name']}_L{i+1}.png", key=f"sav_{idx}_{i}")
                
                if st.button(f"🗑️ Elimina {proj['name']}", key=f"del_{idx}"):
                    st.session_state.saved_projects.pop(idx)
                    st.rerun()

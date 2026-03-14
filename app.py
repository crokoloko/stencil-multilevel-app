import streamlit as st
import cv2
import numpy as np
import zipfile
from io import BytesIO

# ==========================================
# 1. Configurazione e Stile (Tema Bianco Panna)
# ==========================================
st.set_page_config(page_title="Chroma Stencil Lab PRO", layout="centered")

if 'saved_projects' not in st.session_state:
    st.session_state.saved_projects = []

st.markdown("""
<style>
    /* Sfondo Bianco Panna */
    .stApp { 
        background-color: #FFFDD0; 
        color: #1e1e1e; 
    }
    
    h1 { 
        font-family: 'Bungee', cursive; 
        color: #DAA520 !important; 
        text-align: center; 
    }
    
    /* Box Report Tecnico */
    .spray-info-box {
        background-color: #F5F5DC;
        border: 1px solid #E0E0D0;
        border-left: 8px solid #FFD700;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 25px;
    }
    
    /* Pulsanti */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #FFD700, #FFA500);
        color: black !important;
        font-weight: bold;
        border: none;
        padding: 12px;
        border-radius: 10px;
    }

    /* Stile Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #F5F5DC;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: #666;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFD700 !important;
        color: black !important;
    }
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

def generate_zip(project):
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        preview_bgr = cv2.cvtColor(project['preview'], cv2.COLOR_RGB2BGR)
        _, prev_img = cv2.imencode(".png", preview_bgr)
        z.writestr("00_anteprima_finale.png", prev_img.tobytes())
        color_summary = "LISTA COLORI BOMBOLETTE:\n"
        for i, mask in enumerate(project['masks']):
            _, m_img = cv2.imencode(".png", mask)
            filename = f"strato_{i+1}_colore_{project['colors'][i].replace('#','')}.png"
            z.writestr(filename, m_img.tobytes())
            color_summary += f"Strato {i+1}: {project['colors'][i]}\n"
        z.writestr("istruzioni.txt", color_summary + "\nUsa le crocette agli angoli per allineare gli strati.")
    return buf.getvalue()

# ==========================================
# 3. Flusso Applicazione
# ==========================================

st.title("🌈 Chroma Stencil Lab")

tab_editor, tab_saved = st.tabs(["🏗️ EDITOR PROGETTO", "💾 ARCHIVIO SALVATI"])

with tab_editor:
    up_file = st.file_uploader("1. Carica la foto da trasformare", type=["jpg", "png", "jpeg"])
    
    if up_file:
        img = cv2.imdecode(np.frombuffer(up_file.read(), np.uint8), 1)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Foto Originale")
        
        with st.expander("⚙️ Parametri Tecnici", expanded=True):
            c1, c2 = st.columns(2)
            n_layers = c1.slider("Numero strati", 2, 8, 4)
            b_len = c1.slider("Lunghezza ponti", 10, 80, 30)
            b_thick = c2.slider("Spessore linee", 1, 10, 2)
            cross_size = c2.slider("Taglia crocette", 10, 50, 20)

        if st.button("✨ ELABORA STENCIL"):
            # K-Means e Maschere
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
            
            # --- MENU RISULTATI (Qui è tornata l'anteprima strati) ---
            st.header("🏁 Risultati Elaborazione")
            res_tab_prev, res_tab_layers = st.tabs(["🌌 ANTEPRIMA URBAN", "✂️ STRATI DA RITAGLIARE"])
            
            with res_tab_prev:
                preview_img = create_preview(masks, hex_colors)
                st.image(preview_img, use_container_width=True, caption="Simulazione finale")
                
                # Sezione Colori
                st.markdown("### 🖌️ Colori Rilevati")
                col_c = st.columns(n_layers)
                for i in range(n_layers):
                    col_c[i].color_picker(f"L{i+1}", hex_colors[i], key=f"cp_ed_{i}")

                if st.button("💾 SALVA NELL'ARCHIVIO"):
                    project = {
                        "name": f"Stencil_{len(st.session_state.saved_projects)+1}",
                        "preview": preview_img,
                        "masks": masks,
                        "colors": hex_colors
                    }
                    st.session_state.saved_projects.append(project)
                    st.success("Progetto aggiunto all'archivio!")

            with res_tab_layers:
                st.info("Visualizza e controlla ogni strato prima di scaricare.")
                # Sottoschede per ogni strato (molto ordinato)
                layer_tabs = st.tabs([f"Strato {i+1}" for i in range(n_layers)])
                for i, l_tab in enumerate(layer_tabs):
                    with l_tab:
                        col_mask, col_info = st.columns([2, 1])
                        col_mask.image(masks[i], caption=f"Maschera Livello {i+1}", use_container_width=True)
                        with col_info:
                            st.write(f"**Dettagli Strato {i+1}**")
                            st.write(f"Colore: {hex_colors[i]}")
                            _, buf = cv2.imencode(".png", masks[i])
                            st.download_button(f"📥 Scarica PNG {i+1}", buf.tobytes(), f"strato_{i+1}.png", key=f"btn_l_{i}")

# ==========================================
# 4. Sezione Archivio Salvati
# ==========================================
with tab_saved:
    if not st.session_state.saved_projects:
        st.info("L'archivio è vuoto. Crea un progetto nell'Editor.")
    else:
        for idx, proj in enumerate(st.session_state.saved_projects):
            with st.expander(f"📁 {proj['name']} - {len(proj['masks'])} strati"):
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    st.image(proj['preview'], use_container_width=True)
                with col_b:
                    st.write("📦 **Pacchetto Progetto**")
                    zip_data = generate_zip(proj)
                    st.download_button(
                        label="📥 SCARICA TUTTO (.ZIP)",
                        data=zip_data,
                        file_name=f"{proj['name']}.zip",
                        mime="application/zip",
                        key=f"zip_arch_{idx}"
                    )
                    if st.button(f"🗑️ Elimina", key=f"del_arch_{idx}"):
                        st.session_state.saved_projects.pop(idx)
                        st.rerun()

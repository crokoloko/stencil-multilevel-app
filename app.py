import streamlit as st
import cv2
import numpy as np

# ==========================================
# 1. Stile UI (Pulito & Moderno)
# ==========================================
st.set_page_config(page_title="Urban Stencil Lab", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    h1 { font-family: 'Bungee', cursive; color: #FFD700 !important; text-align: center; margin-bottom: 30px; }
    
    /* Box Info Bomboletta */
    .spray-info-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-left: 8px solid #FFD700;
        padding: 20px;
        border-radius: 8px;
        margin: 20px 0;
    }
    
    /* Stile Pulsante Genera */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #FFD700, #FFA500);
        color: black !important;
        font-weight: bold;
        border: none;
        padding: 15px;
        font-size: 1.2rem;
        border-radius: 10px;
        transition: 0.3s;
    }
    .stButton>button:hover { transform: scale(1.01); box-shadow: 0 0 20px rgba(255, 215, 0, 0.4); }

    /* Score labels */
    .score-val { font-size: 1.8rem; font-weight: 900; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Funzioni Core
# ==========================================

def get_stencil_analytics(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Contrasto
    contrast = np.std(gray)
    # Complessità (Bordi)
    edges = cv2.Canny(gray, 100, 200)
    density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
    
    score = int(np.clip((contrast/60)*50 + (30 - density*2) + 20, 0, 100))
    rec_layers = 3 if score < 50 else (4 if score < 80 else 5)
    
    status = "ECCELLENTE 🚀" if score > 75 else ("BUONA 🎨" if score > 50 else "DIFFICILE ⚠️")
    color = "#00FF00" if score > 75 else ("#FFA500" if score > 50 else "#FF0000")
    
    return score, status, color, rec_layers

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

# ==========================================
# 3. Flusso Applicazione (Scroll Ordinato)
# ==========================================

st.title("🌈 Chroma Stencil Lab")

# STEP 1: CARICAMENTO
up_file = st.file_uploader("1. Inizia caricando la tua immagine", type=["jpg", "png", "jpeg"])

if up_file:
    # Lettura Immagine
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # STEP 2: ANALISI (Info Box)
    score, status, s_color, rec_layers = get_stencil_analytics(img)
    
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    st.markdown(f"""
    <div class="spray-info-box">
        <h4 style="margin:0;">📓 REPORT TECNICO</h4>
        <p style="margin:10px 0;">Adattabilità: <span class="score-val" style="color:{s_color};">{score}/100</span> — <b>{status}</b></p>
        <p style="font-size:0.9rem; color:#8b949e;">Consigliamo <b>{rec_layers} strati</b> per questo tipo di dettaglio.</p>
    </div>
    """, unsafe_allow_html=True)

    # STEP 3: CONFIGURAZIONE
    st.header("⚙️ Configura lo Stencil")
    c1, c2 = st.columns(2)
    with c1:
        n_layers = st.slider("Numero di strati", 2, 8, rec_layers)
        b_len = st.slider("Lunghezza ponti", 10, 80, 30)
    with c2:
        b_thick = st.slider("Spessore linee", 1, 10, 2)
        cross_size = st.slider("Taglia crocette", 10, 50, 20)

    st.markdown("---")

    # STEP 4: AZIONE GENERATORE
    if st.button("✨ GENERA ANTEPRIMA E STRATI"):
        # Processing K-Means
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        data = img_lab.reshape((-1, 3)).astype(np.float32)
        _, label, centers = cv2.kmeans(data, n_layers, None, (cv2.TERM_CRITERIA_EPS+20, 20, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        quantized_lab = centers[label.flatten()].reshape((img_lab.shape))

        # Generazione Maschere e Colori
        masks = []
        original_hex_colors = []
        for i in range(n_layers):
            m = cv2.inRange(quantized_lab, centers[i], centers[i])
            masks.append(apply_bridges_and_crosses(m, b_len, b_thick, cross_size))
            rgb = cv2.cvtColor(np.uint8([[centers[i]]]), cv2.COLOR_LAB2RGB)[0][0]
            original_hex_colors.append('#%02x%02x%02x' % tuple(rgb))

        # --- RISULTATI: ANTEPRIMA URBAN ---
        st.header("🌌 Anteprima sul Muro")
        h, w = masks[0].shape
        # Creazione texture cemento
        bg = np.full((h, w, 3), 100, dtype=np.uint8)
        noise = np.random.normal(0, 15, (h, w, 3)).astype(np.int16)
        canvas = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        for i in range(n_layers):
            m = masks[i]
            rgb = tuple(int(original_hex_colors[i].lstrip('#')[j:j+2], 16) for j in (0, 2, 4))
            bgr = (rgb[2], rgb[1], rgb[0])
            color_img = np.full((h, w, 3), bgr, dtype=np.uint8)
            layer_c = cv2.bitwise_and(color_img, color_img, mask=m)
            bg_p = cv2.bitwise_and(canvas, canvas, mask=m)
            blended = cv2.addWeighted(bg_p, 0.4, layer_c, 0.6, 0)
            canvas = cv2.bitwise_and(canvas, canvas, mask=cv2.bitwise_not(m))
            canvas = cv2.add(canvas, blended)
        
        st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), use_container_width=True)

        # --- STEP 5: PERSONALIZZAZIONE TINTA E DOWNLOAD (In fondo) ---
        st.markdown("---")
        st.header("🖌️ Personalizza Tinte e Scarica")
        st.info("Scegli i colori delle tue bombolette spray per ogni strato qui sotto.")
        
        # Menu a schede per non affollare
        tabs = st.tabs([f"Strato {i+1}" for i in range(n_layers)])
        for i, tab in enumerate(tabs):
            with tab:
                col_view, col_ctrl = st.columns([2, 1])
                with col_view:
                    st.image(masks[i], caption=f"Maschera di Taglio {i+1}", use_container_width=True)
                with col_ctrl:
                    st.color_picker(f"Colore Bomboletta {i+1}", original_hex_colors[i], key=f"p_{i}")
                    st.write(f"Intensità originale: {original_hex_colors[i]}")
                    _, buf = cv2.imencode(".png", masks[i])
                    st.download_button(f"📥 Scarica PNG {i+1}", buf.tobytes(), f"stencil_layer_{i+1}.png", "image/png")

st.markdown("<br><br><p style='text-align:center; color:#444;'>Urban Stencil Lab v2.0</p>", unsafe_allow_html=True)

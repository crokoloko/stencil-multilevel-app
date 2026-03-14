import streamlit as st
import cv2
import numpy as np

# ==========================================
# 1. Stile UI (Menu e Box Bomboletta)
# ==========================================
st.set_page_config(page_title="Stencilability Lab", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    h1 { font-family: 'Bungee', cursive; color: #FFD700 !important; text-align: center; }
    
    /* Box stile Bomboletta / Score */
    .spray-info-box {
        background-color: #1e1e1e;
        border-left: 10px solid #FFD700;
        padding: 20px;
        border-radius: 0px 20px 20px 0px;
        font-family: 'Courier New', Courier, monospace;
        color: #00FFD1;
        box-shadow: 5px 5px 15px rgba(0,0,0,0.5);
    }
    .score-high { color: #00FF00; font-weight: bold; font-size: 1.5rem; }
    .score-med { color: #FFA500; font-weight: bold; font-size: 1.5rem; }
    .score-low { color: #FF0000; font-weight: bold; font-size: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Funzioni di Analisi Avanzata
# ==========================================

def calculate_stencil_score(img):
    """Calcola quanto la foto è adatta agli stencil (0-100)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Analisi del Contrasto (Deviazione Standard)
    contrast = np.std(gray)
    contrast_score = np.clip((contrast / 60) * 40, 0, 40) # Max 40 punti
    
    # 2. Analisi della Complessità (Bordi/Rumore)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
    # Più i bordi sono densi, più è difficile ritagliare (troppo dettaglio = male)
    complexity_score = np.clip(30 - (edge_density * 2), 0, 30) # Max 30 punti
    
    # 3. Analisi della Saturazione (Per stencil a colori)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = np.mean(hsv[:,:,1])
    sat_score = np.clip((saturation / 100) * 30, 0, 30) # Max 30 punti
    
    total_score = int(contrast_score + complexity_score + sat_score)
    return total_score

def get_image_info(img):
    small_img = cv2.resize(img, (150, 150))
    pixels = small_img.reshape(-1, 3)
    unique_colors = len(np.unique(pixels, axis=0))
    score = calculate_stencil_score(img)
    
    if score > 75: 
        label, cls = "ECCELLENTE 🚀", "score-high"
    elif score > 50: 
        label, cls = "BUONA 🎨", "score-med"
    else: 
        label, cls = "DIFFICILE ⚠️", "score-low"
        
    return unique_colors, label, cls, score

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
# 3. Interfaccia
# ==========================================

st.title("🌈 Chroma Stencil Lab")

up_file = st.file_uploader("Carica la foto", type=["jpg", "png", "jpeg"])

if up_file:
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Analisi Avanzata
    n_unique, score_label, score_class, score_num = get_image_info(img)
    
    col_img, col_info = st.columns([1, 1])
    with col_img:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Originale", use_container_width=True)
    with col_info:
        st.markdown(f"""
        <div class="spray-info-box">
            <h4>📓 STENCILABILITY REPORT</h4>
            <p><b>Punteggio Adattabilità:</b> <span class="{score_class}">{score_num}/100</span></p>
            <p><b>Giudizio:</b> <span class="{score_class}">{score_label}</span></p>
            <hr style="border-color: #444;">
            <p style="font-size: 0.9rem;"><b>Dettagli Tecnici:</b><br>
            La foto presenta {n_unique:,} tonalità. 
            {"Il contrasto è ottimo per separare le masse." if score_num > 60 else "Potrebbe servire più contrasto prima di procedere."}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Parametri
    c1, c2, c3, c4 = st.columns(4)
    n_colors = c1.slider("Numero Strati", 2, 8, 4)
    b_len = c2.slider("Lunghezza Ponti", 10, 80, 30)
    b_thick = c3.slider("Spessore Linee", 1, 10, 2)
    cross_size = c4.slider("Taglia Crocette", 10, 50, 20)

    # Processing K-Means
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    data = img_lab.reshape((-1, 3)).astype(np.float32)
    _, label, centers = cv2.kmeans(data, n_colors, None, (cv2.TERM_CRITERIA_EPS+20, 20, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    quantized_lab = centers[label.flatten()].reshape((img_lab.shape))

    masks = []
    cols_hex = []
    for i in range(n_colors):
        m = cv2.inRange(quantized_lab, centers[i], centers[i])
        masks.append(apply_bridges_and_crosses(m, b_len, b_thick, cross_size))
        rgb = cv2.cvtColor(np.uint8([[centers[i]]]), cv2.COLOR_LAB2RGB)[0][0]
        cols_hex.append('#%02x%02x%02x' % tuple(rgb))

    # MENU A TAB
    st.subheader("✂️ Esplora i Livelli")
    tab_names = [f"Strato {i+1}" for i in range(n_colors)]
    tabs = st.tabs(tab_names)

    for i, tab in enumerate(tabs):
        with tab:
            col_t1, col_t2 = st.columns([2, 1])
            with col_t1:
                st.image(masks[i], caption=f"Maschera di Taglio {i+1}", use_container_width=True)
            with col_t2:
                st.markdown(f"**Colore rilevato:**")
                st.color_picker(f"Tinta {i+1}", cols_hex[i], key=f"cp_{i}")
                _, buf = cv2.imencode(".png", masks[i])
                st.download_button(f"📥 Scarica PNG {i+1}", buf.tobytes(), f"layer_{i+1}.png", "image/png")

    st.markdown("---")
    
    # Simulatore Finale
    if st.button("✨ GENERA ANTEPRIMA URBAN (CEMENTO)"):
        h, w = masks[0].shape
        # Creazione texture cemento
        bg = np.full((h, w, 3), 100, dtype=np.uint8)
        noise = np.random.normal(0, 15, (h, w, 3)).astype(np.int16)
        canvas = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        for i in range(n_colors):
            m = masks[i]
            rgb = tuple(int(cols_hex[i].lstrip('#')[j:j+2], 16) for j in (0, 2, 4))
            bgr = (rgb[2], rgb[1], rgb[0])
            color_img = np.full((h, w, 3), bgr, dtype=np.uint8)
            layer_c = cv2.bitwise_and(color_img, color_img, mask=m)
            bg_p = cv2.bitwise_and(canvas, canvas, mask=m)
            blended = cv2.addWeighted(bg_p, 0.4, layer_c, 0.6, 0)
            canvas = cv2.bitwise_and(canvas, canvas, mask=cv2.bitwise_not(m))
            canvas = cv2.add(canvas, blended)
        st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), use_container_width=True)

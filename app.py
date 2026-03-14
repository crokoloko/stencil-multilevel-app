import streamlit as st
import cv2
import numpy as np

# ==========================================
# 1. Configurazione e Stile (Spray Can Look)
# ==========================================
st.set_page_config(page_title="Chroma Stencil Lab", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    h1 { font-family: 'Bungee', cursive; color: #FFD700 !important; text-align: center; }
    
    /* Box stile Bomboletta / Industrial */
    .spray-info-box {
        background-color: #1e1e1e;
        border-left: 10px solid #FFD700;
        border-right: 2px solid #333;
        border-top: 2px solid #333;
        border-bottom: 2px solid #333;
        padding: 20px;
        border-radius: 0px 20px 20px 0px;
        font-family: 'Courier New', Courier, monospace;
        color: #00FFD1;
        box-shadow: 5px 5px 15px rgba(0,0,0,0.5);
        margin-top: 10px;
    }
    .spray-info-box h4 { color: #FFD700 !important; margin-bottom: 10px; text-transform: uppercase; }
    .info-label { color: #888; font-weight: bold; }
    
    .stButton>button { width: 100%; background: #FFD700; color: black; font-weight: bold; border-radius: 10px; border: none; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Funzioni di Analisi e Disegno
# ==========================================

def get_image_info(img):
    """Analizza l'immagine per estrarre info sui colori."""
    # Riduciamo l'immagine per velocizzare l'analisi dei colori unici
    small_img = cv2.resize(img, (150, 150))
    pixels = small_img.reshape(-1, 3)
    unique_colors = len(np.unique(pixels, axis=0))
    
    # Calcolo strati consigliati (logica basata sulla complessità cromatica)
    if unique_colors < 5000:
        recommended = 3
    elif unique_colors < 15000:
        recommended = 4
    elif unique_colors < 30000:
        recommended = 5
    else:
        recommended = 6
        
    return unique_colors, recommended

def create_concrete_texture(height, width):
    bg = np.full((height, width, 3), 100, dtype=np.uint8)
    noise = np.random.normal(0, 12, (height, width, 3)).astype(np.int16)
    concrete = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(concrete, (3, 3), 0)

def apply_bridges(mask, b_len, b_thick):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = mask.copy()
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, cnt in enumerate(contours):
            if hierarchy[i][3] != -1 and cv2.contourArea(cnt) > 80:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                    # Disegna il ponte bianco (interrompe il taglio nero)
                    cv2.line(out, (cX, cY), (cX, cY - b_len), 255, b_thick)
    return out

# --- NUOVA FUNZIONE: Disegna Crocette ---
def draw_alignment_crosses(mask, size, thickness):
    """Aggiunge crocette di allineamento agli angoli."""
    h, w = mask.shape
    out = mask.copy()
    
    # Definiamo le posizioni delle crocette (negli angoli)
    margin = size + thickness + 10 # Margine dai bordi
    centers = [
        (margin, margin),             # Alto a sinistra
        (w - margin, margin),         # Alto a destra
        (margin, h - margin),         # Basso a sinistra
        (w - margin, h - margin)      # Basso a destra
    ]
    
    for cX, cY in centers:
        # Disegna la linea orizzontale
        cv2.line(out, (cX - size, cY), (cX + size, cY), 255, thickness)
        # Disegna la linea verticale
        cv2.line(out, (cX, cY - size), (cX, cY + size), 255, thickness)
        
    return out

# ==========================================
# 3. Interfaccia Principale
# ==========================================

st.title("🌈 Chroma Stencil Lab")

up_file = st.file_uploader("Carica la tua foto", type=["jpg", "png", "jpeg"])

if up_file:
    # Caricamento e analisi
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Analisi immagine
    n_unique, rec_layers = get_image_info(img)

    col_img, col_info = st.columns([1, 1])
    
    with col_img:
        st.image(img_rgb, caption="Anteprima Originale", use_container_width=True)
    
    with col_info:
        # --- BOX IN STILE BOMBOLETTA ---
        st.markdown(f"""
        <div class="spray-info-box">
            <h4>📓 ANALISI BOMBOLETTA (Spec)</h4>
            <p><span class="info-label">Dimensioni:</span> {img.shape[1]}x{img.shape[0]} px</p>
            <p><span class="info-label">Varietà Cromatica:</span> {n_unique:,} tonalità rilevate</p>
            <hr style="border-color: #333;">
            <p><span class="info-label">CONSIGLIO PRO:</span></p>
            <p>Per un realismo ottimale, si consiglia di utilizzare <b>{rec_layers} strati</b> di colore.</p>
            <p style="font-size: 0.8rem; color: #666;">*Basato sulla densità dei gradienti rilevati.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Parametri
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        n_colors = st.slider("Numero Strati Scelti", 2, 8, rec_layers)
    with col_p2:
        b_len = st.slider("Lunghezza Ponti", 10, 100, 30)
    with col_p3:
        b_thick = st.slider("Spessore Ponti", 1, 10, 3)
    with col_p4:
        cross_size = st.slider("Dimensione Crocette", 10, 50, 20)

    # --- PROCESSING ---
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    data = img_lab.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, label, centers = cv2.kmeans(data, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    res = centers[label.flatten()]
    quantized_lab = res.reshape((img_lab.shape))

    stencil_masks = []
    extracted_colors = []
    
    for i in range(n_colors):
        color_target = centers[i]
        mask = cv2.inRange(quantized_lab, color_target, color_target)
        
        # 1. Applica ponti
        mask_with_bridges = apply_bridges(mask, b_len, b_thick)
        
        # 2. Applica crocette (usiamo lo stesso spessore dei ponti)
        final_mask = draw_alignment_crosses(mask_with_bridges, cross_size, b_thick)
        
        stencil_masks.append(final_mask)
        c_bgr = cv2.cvtColor(np.uint8([[color_target]]), cv2.COLOR_LAB2RGB)[0][0]
        extracted_colors.append('#%02x%02x%02x' % tuple(c_bgr))

    # --- SIMULATORE ---
    st.markdown("---")
    if st.button("✨ GENERA ANTEPRIMA SUL MURO"):
        h, w = stencil_masks[0].shape
        canvas = create_concrete_texture(h, w)
        for i in range(n_colors):
            mask = stencil_masks[i]
            # Converti Hex to RGB to BGR per OpenCV
            rgb = tuple(int(extracted_colors[i].lstrip('#')[j:j+2], 16) for j in (0, 2, 4))
            bgr = (rgb[2], rgb[1], rgb[0])
            color_img = np.full((h, w, 3), bgr, dtype=np.uint8)
            layer_colored = cv2.bitwise_and(color_img, color_img, mask=mask)
            bg_part = cv2.bitwise_and(canvas, canvas, mask=mask)
            blended = cv2.addWeighted(bg_part, 0.4, layer_colored, 0.6, 0)
            canvas = cv2.bitwise_and(canvas, canvas, mask=cv2.bitwise_not(mask))
            canvas = cv2.add(canvas, blended)
        st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), use_container_width=True)

    # --- DOWNLOAD ---
    st.subheader("📥 Download Strati")
    dl_cols = st.columns(n_colors)
    for i in range(n_colors):
        with dl_cols[i]:
            # Mostra l'immagine dello strato con crocette
            st.image(stencil_masks[i], caption=f"Strato {i+1}", use_container_width=True)
            
            # Bottone Download
            _, buf = cv2.imencode(".png", stencil_masks[i])
            st.download_button(f"PNG {i+1}", buf.tobytes(), f"layer_{i+1}.png", "image/png", key=f"dl_{i}")

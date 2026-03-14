import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ==========================================
# 1. Stile UI (Mix Pop & Urban)
# ==========================================
st.set_page_config(page_title="Urban Stencil Lab", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #1a1a1a; color: #ffffff; }
    h1, h2 { font-family: 'Bungee', cursive; color: #00FFD1 !important; text-align: center; text-shadow: 2px 2px #FF00FF; }
    .stButton>button { width: 100%; background: linear-gradient(45deg, #FF00FF, #00FFD1); border: none; color: white; font-weight: bold; border-radius: 10px; }
    [data-testid="stFileUploader"] { border: 2px dashed #00FFD1; background: #262626; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Funzioni Core & Texture
# ==========================================

def create_concrete_texture(height, width):
    """Genera una texture di cemento realistica usando il rumore."""
    # Base grigia media
    bg = np.full((height, width, 3), 120, dtype=np.uint8)
    # Aggiunta di grana (rumore)
    noise = np.random.normal(0, 15, (height, width, 3)).astype(np.int16)
    concrete = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Aggiunta di macchie più grandi (nuvole di fumo) per realismo
    concrete = cv2.GaussianBlur(concrete, (5, 5), 0)
    return concrete

def apply_bridges(mask, b_len, b_thick):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = mask.copy()
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, cnt in enumerate(contours):
            if hierarchy[i][3] != -1 and cv2.contourArea(cnt) > 100:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                    cv2.line(out, (cX, cY), (cX, cY - b_len), 255, b_thick)
    return out

def simulate_urban_result(layers, colors, num_layers):
    h, w = layers[0].shape
    # Creiamo il muro di cemento
    canvas = create_concrete_texture(h, w)
    
    for i in range(num_layers):
        mask = layers[i]
        c_hex = colors[i]
        # Converti Hex to BGR
        c_bgr = tuple(int(c_hex.lstrip('#')[j:j+2], 16) for j in (4, 2, 0))
        
        # Crea lo strato di colore
        color_img = np.full((h, w, 3), c_bgr, dtype=np.uint8)
        
        # Applichiamo il colore solo dove c'è la maschera
        # Usiamo un effetto "spray": lasciamo intravedere un po' della grana del muro sotto
        layer_rgb = cv2.bitwise_and(color_img, color_img, mask=mask)
        
        # Sovrapposizione con trasparenza per simulare l'assorbimento della vernice
        alpha = 0.85
        mask_3d = cv2.merge([mask, mask, mask])
        inv_mask = cv2.bitwise_not(mask_3d)
        
        # Parte del muro che viene coperta
        bg_part = cv2.bitwise_and(canvas, canvas, mask=mask)
        # Mix tra colore e grana del muro
        blended_part = cv2.addWeighted(bg_part, 0.3, layer_rgb, 0.7, 0)
        # Ricomponiamo l'immagine
        canvas = cv2.bitwise_and(canvas, canvas, mask=cv2.bitwise_not(mask))
        canvas = cv2.add(canvas, blended_part)
        
    return canvas

# ==========================================
# 3. Interfaccia UI
# ==========================================

st.title("🏙️ Urban Stencil Simulator")

# Layout principale
col_input, col_config = st.columns([1, 1])

with col_input:
    up_file = st.file_uploader("Carica immagine", type=["jpg", "png"])
with col_config:
    n_layers = st.slider("Livelli", 2, 6, 3)
    b_len = st.slider("Ponti (Lunghezza)", 10, 80, 30)
    b_thick = st.slider("Ponti (Spessore)", 1, 10, 2)

if up_file:
    # Processing
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # K-Means logic
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray, 9, 75, 75)
    data = smooth.reshape((-1, 1)).astype(np.float32)
    _, label, centers = cv2.kmeans(data, n_layers, None, (cv2.TERM_CRITERIA_EPS+10, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(np.sort(centers.flatten()))
    quantized = centers[label.flatten()].reshape(smooth.shape)

    st.markdown("---")
    
    # Generazione Strati e Colori
    stencil_layers = []
    color_picks = []
    
    st.subheader("🎨 Configura Colori e Anteprima")
    c_pick, c_prev = st.columns([1, 2])
    
    with c_pick:
        for i in range(n_layers):
            mask = cv2.inRange(quantized, int(centers[i]), int(centers[i]))
            final_mask = apply_bridges(mask, b_len, b_thick)
            stencil_layers.append(final_mask)
            
            # Default colors: dal nero al bianco
            def_col = f"#{i*40:02x}{i*40:02x}{i*40:02x}"
            color_picks.append(st.color_picker(f"Colore Strato {i+1}", def_col))

    with c_prev:
        if st.button("🚀 GENERA ANTEPRIMA SUL MURO"):
            res = simulate_urban_result(stencil_layers, color_picks, n_layers)
            st.image(cv2.cvtColor(res, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Anteprima su Cemento")

    st.markdown("---")
    st.subheader("📥 Scarica i tuoi Stencil")
    dl_cols = st.columns(n_layers)
    for i in range(n_layers):
        with dl_cols[i]:
            st.image(stencil_layers[i], caption=f"Strato {i+1}")
            _, buf = cv2.imencode(".png", stencil_layers[i])
            st.download_button(f"PNG {i+1}", buf.tobytes(), f"layer_{i+1}.png", "image/png")

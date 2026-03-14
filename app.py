import streamlit as st
import cv2
import numpy as np

# ==========================================
# 1. Configurazione e Stile
# ==========================================
st.set_page_config(page_title="Chroma Stencil Lab", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    h1 { font-family: 'Bungee', cursive; color: #FFD700 !important; text-align: center; }
    .stButton>button { width: 100%; background: #FFD700; color: black; font-weight: bold; border-radius: 10px; border: none; }
    .stButton>button:hover { background: #FFEA00; color: black; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Logica di Elaborazione Colore
# ==========================================

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
                    cv2.line(out, (cX, cY), (cX, cY - b_len), 255, b_thick)
    return out

# ==========================================
# 3. Interfaccia Principale
# ==========================================

st.title("🌈 Chroma Stencil Lab")
st.write("Dividi l'immagine in base ai colori reali per stencil multi-tonali.")

up_file = st.file_uploader("Carica la tua opera", type=["jpg", "png", "jpeg"])

if up_file:
    # Parametri in cima
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        n_colors = st.slider("Quanti colori (strati)?", 2, 8, 4)
    with col_p2:
        b_len = st.slider("Lunghezza Ponti", 10, 100, 30)
    with col_p3:
        b_thick = st.slider("Spessore Ponti", 1, 10, 3)

    # Lettura immagine
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    st.image(img_rgb, width=400, caption="Originale")

    # --- ALGORITMO DI SEPARAZIONE COLORI ---
    # Usiamo lo spazio colore LAB per una separazione cromatica migliore
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    data = img_lab.reshape((-1, 3)).astype(np.float32)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, label, centers = cv2.kmeans(data, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    res = centers[label.flatten()]
    quantized_lab = res.reshape((img_lab.shape))
    quantized_bgr = cv2.cvtColor(quantized_lab, cv2.COLOR_LAB2BGR)

    st.markdown("---")
    
    # Generazione Strati
    stencil_masks = []
    extracted_colors = []
    
    for i in range(n_colors):
        # Isola il colore specifico trovato da K-Means
        color_target = centers[i]
        mask = cv2.inRange(quantized_lab, color_target, color_target)
        
        # Applica ponti alla maschera
        final_mask = apply_bridges(mask, b_len, b_thick)
        stencil_masks.append(final_mask)
        
        # Converti il colore del centro LAB in HEX per la UI
        c_bgr = cv2.cvtColor(np.uint8([[color_target]]), cv2.COLOR_LAB2RGB)[0][0]
        hex_col = '#%02x%02x%02x' % tuple(c_bgr)
        extracted_colors.append(hex_col)

    # --- SIMULATORE E DOWNLOAD ---
    col_sim, col_list = st.columns([2, 1])
    
    with col_list:
        st.subheader("🎨 Colori Rilevati")
        final_colors = []
        for i in range(n_colors):
            c = st.color_picker(f"Strato {i+1}", extracted_colors[i], key=f"cp_{i}")
            final_colors.append(c)

    with col_sim:
        if st.button("✨ GENERA ANTEPRIMA URBAN"):
            # Simulazione su cemento
            h, w = stencil_masks[0].shape
            canvas = create_concrete_texture(h, w)
            
            for i in range(n_colors):
                mask = stencil_masks[i]
                c_hex = final_colors[i]
                rgb = tuple(int(c_hex.lstrip('#')[j:j+2], 16) for j in (0, 2, 4))
                bgr = (rgb[2], rgb[1], rgb[0])
                
                # Effetto spray
                color_img = np.full((h, w, 3), bgr, dtype=np.uint8)
                layer_colored = cv2.bitwise_and(color_img, color_img, mask=mask)
                
                # Overlay
                bg_part = cv2.bitwise_and(canvas, canvas, mask=mask)
                blended = cv2.addWeighted(bg_part, 0.4, layer_colored, 0.6, 0)
                canvas = cv2.bitwise_and(canvas, canvas, mask=cv2.bitwise_not(mask))
                canvas = cv2.add(canvas, blended)
            
            st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.markdown("---")
    st.subheader("📥 Download Stencil")
    dl_cols = st.columns(n_colors)
    for i in range(n_colors):
        with dl_cols[i]:
            st.image(stencil_masks[i], caption=f"Taglio {i+1}")
            _, buf = cv2.imencode(".png", stencil_masks[i])
            st.download_button(f"Scarica {i+1}", buf.tobytes(), f"color_layer_{i+1}.png", "image/png")

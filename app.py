import streamlit as st
import cv2
import numpy as np
from PIL import Image

def apply_bridges(mask, bridge_length, thickness):
    # Analisi dei contorni con gerarchia
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    output = mask.copy()
    
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, cnt in enumerate(contours):
            # Se ha un padre (hierarchy[i][3] != -1), è un'isola/buco
            if hierarchy[i][3] != -1:
                area = cv2.contourArea(cnt)
                if area > 100: # Filtro dimensione minima
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        # Disegna il ponte verso l'alto (bianco su maschera nera)
                        cv2.line(output, (cX, cY), (cX, cY - bridge_length), 255, thickness)
    return output

st.set_page_config(page_title="Stencil Maker AI", layout="wide")
st.title("🎨 Stencil Layer Generator")
st.write("Carica una foto e genera i livelli per i tuoi stencil con ponti automatici.")

# Sidebar per i parametri
st.sidebar.header("Impostazioni Stencil")
num_layers = st.sidebar.slider("Numero di livelli", 2, 6, 3)
bridge_len = st.sidebar.slider("Lunghezza Ponti (px)", 10, 100, 30)
bridge_thick = st.sidebar.slider("Spessore Ponti (px)", 1, 10, 3)

uploaded_file = st.file_uploader("Scegli un'immagine...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Conversione immagine per OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_rgb, caption="Immagine Originale", use_column_width=True)

    # Elaborazione
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # K-Means Posterization
    data = smooth.reshape((-1, 1)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, centers = cv2.kmeans(data, num_layers, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(np.sort(centers.flatten()))
    quantized = centers[label.flatten()].reshape(smooth.shape)

    st.subheader("Livelli Generati")
    cols = st.columns(num_layers)
    
    for i in range(num_layers):
        # Isola lo strato
        layer_mask = cv2.inRange(quantized, int(centers[i]), int(centers[i]))
        # Applica i ponti
        final_layer = apply_bridges(layer_mask, bridge_len, bridge_thick)
        
        with cols[i]:
            st.image(final_layer, caption=f"Strato {i+1} (Grigio: {centers[i]})", use_column_width=True)
            # Bottone per scaricare il singolo livello
            res, thumb = cv2.imencode(".png", final_layer)
            st.download_button(f"Scarica Strato {i+1}", data=thumb.tobytes(), file_name=f"stencil_layer_{i+1}.png", mime="image/png")

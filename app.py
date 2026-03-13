import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ==========================================
# 1. Configurazione e Stile Space & Neon
# ==========================================
st.set_page_config(page_title="Stencil Space Lab", layout="centered")

space_css = """
<style>
    /* Sfondo nero stellato (CSS puro) */
    .stApp {
        background-color: #000000;
        background-image: 
            radial-gradient(white, rgba(255,255,255,.2) 2px, transparent 40px),
            radial-gradient(white, rgba(255,255,255,.15) 1px, transparent 30px),
            radial-gradient(white, rgba(255,255,255,.1) 2px, transparent 40px);
        background-size: 550px 550px, 350px 350px, 250px 250px;
        color: #00FFD1; /* Turchese Neon */
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }

    /* Rimuovi Sidebar e Menu Streamlit */
    [data-testid="stSidebar"], section[data-testid="stSidebarNav"] {
        display: none;
    }

    /* Titoli Neon */
    h1, h2, h3 {
        color: #FF00FF !important; /* Magenta Neon */
        text-shadow: 0 0 10px #FF00FF, 0 0 20px #FF00FF;
        text-align: center;
        text-transform: uppercase;
    }

    /* Card per le impostazioni */
    .stSlider, .stFileUploader {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #00FFD1;
        margin-bottom: 20px;
    }

    /* Pulsante Genera */
    .stButton>button {
        width: 100%;
        background: linear-gradient(45deg, #FF00FF, #00FFD1) !important;
        color: white !important;
        font-weight: bold !important;
        font-size: 1.5rem !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 15px !important;
        box-shadow: 0 0 20px rgba(0, 255, 209, 0.4);
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 40px rgba(255, 0, 255, 0.6);
    }

    /* Testi etichette */
    label {
        color: #00FFD1 !important;
        font-size: 1.1rem !important;
    }
</style>
"""
st.markdown(space_css, unsafe_allow_html=True)

# ==========================================
# 2. Logica di Elaborazione
# ==========================================

def process_stencil(img, layers_count, b_len, b_thick):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # K-Means
    data = smooth.reshape((-1, 1)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, centers = cv2.kmeans(data, layers_count, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(np.sort(centers.flatten()))
    quantized = centers[label.flatten()].reshape(smooth.shape)
    
    results = []
    for i in range(layers_count):
        mask = cv2.inRange(quantized, int(centers[i]), int(centers[i]))
        
        # Ponti
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        out_mask = mask.copy()
        if hierarchy is not None:
            hierarchy = hierarchy[0]
            for j, cnt in enumerate(contours):
                if hierarchy[j][3] != -1 and cv2.contourArea(cnt) > 100:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                        cv2.line(out_mask, (cX, cY), (cX, cY - b_len), 255, b_thick)
        results.append((out_mask, centers[i]))
    return results

# ==========================================
# 3. Layout a Scorrimento
# ==========================================

st.title("🚀 Stencil Galaxy Lab")
st.write("Configura la tua missione e trasforma le immagini in stencil stellari.")

st.markdown("### 🛠️ Configurazione")
num_layers = st.slider("Quanti strati vuoi creare?", 2, 8, 3)
bridge_len = st.slider("Lunghezza ponti di supporto", 10, 150, 40)
bridge_thick = st.slider("Spessore dei ponti", 1, 15, 3)

st.markdown("### 📁 Caricamento")
uploaded_file = st.file_uploader("Scegli il simulacro da processare", type=["jpg", "jpeg", "png"])

# Variabile di stato per la generazione
if 'process' not in st.session_state:
    st.session_state.process = False

if uploaded_file:
    st.image(uploaded_file, caption="Immagine pronta per il decollo", width=300)
    
    st.markdown("---")
    if st.button("✨ CREA STENCIL ORA"):
        st.session_state.process = True

    if st.session_state.process:
        # Elaborazione
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        layers = process_stencil(img, num_layers, bridge_len, bridge_thick)
        
        st.header("🌌 Risultati dell'Elaborazione")
        
        for idx, (l_img, val) in enumerate(layers):
            st.subheader(f"Strato {idx+1} - Intensità: {val}")
            st.image(l_img, use_container_width=True)
            
            # Download
            res, thumb = cv2.imencode(".png", l_img)
            st.download_button(
                label=f"💾 Scarica Strato {idx+1}",
                data=thumb.tobytes(),
                file_name=f"stencil_layer_{idx+1}.png",
                mime="image/png",
                key=f"btn_{idx}"
            )
        
        st.success("Tutti gli strati sono stati forgiati con successo!")

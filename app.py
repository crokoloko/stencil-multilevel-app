import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ==========================================
# 1. Configurazione e Stile "Candy Pop" (CSS)
# ==========================================
st.set_page_config(
    page_title="Stencil Pop Art Maker",
    layout="wide",
)

# --- CSS PERSONALIZZATO (Stile Allegro e Vibrante) ---
pop_art_css = """
<style>
    /* 1. Sfondo e Testo */
    .stApp {
        background-color: #FFDE59; /* Giallo brillante */
        color: #333333;
        font-family: 'Comic Sans MS', 'Chalkboard SE', sans-serif;
    }

    /* 2. Titoli (Stile Bold & Fun) */
    @import url('https://fonts.googleapis.com/css2?family=Bungee&display=swap');
    
    h1, h2, h3 {
        font-family: 'Bungee', cursive !important;
        color: #FF5757 !important; /* Rosso corallo */
        text-shadow: 3px 3px 0px #5271FF; /* Ombra azzurra "pop" */
        text-align: center;
    }

    /* 3. Sidebar */
    [data-testid="stSidebar"] {
        background-color: #5271FF !important; /* Blu elettrico */
        border-right: 5px solid #000000;
    }
    
    [data-testid="stSidebar"] .stMarkdown h2, 
    [data-testid="stSidebar"] label {
        color: #FFFFFF !important;
        font-family: 'Bungee', cursive;
    }

    /* 4. Pulsanti */
    .stButton>button {
        background-color: #8C52FF !important; /* Viola acceso */
        color: #FFFFFF !important;
        border: 3px solid #000000 !important;
        border-radius: 15px !important;
        font-weight: bold !important;
        font-size: 1.2rem !important;
        box-shadow: 5px 5px 0px #000000;
        transition: transform 0.1s;
    }
    .stButton>button:hover {
        transform: translate(-2px, -2px);
        box-shadow: 7px 7px 0px #000000;
        background-color: #FF66C4 !important; /* Rosa shocking */
    }

    /* 5. Uploader */
    [data-testid="stFileUploader"] {
        background-color: #FFFFFF;
        border: 4px dashed #FF66C4 !important;
        border-radius: 20px;
        padding: 20px;
    }
</style>
"""
st.markdown(pop_art_css, unsafe_allow_html=True)

# ==========================================
# 2. Logica Applicativa (Invariata ma Solida)
# ==========================================

def apply_bridges(mask, bridge_length, thickness):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    output = mask.copy()
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, cnt in enumerate(contours):
            if hierarchy[i][3] != -1: # È un buco (isola)
                if cv2.contourArea(cnt) > 100:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                        cv2.line(output, (cX, cY), (cX, cY - bridge_length), 255, thickness)
    return output

# ==========================================
# 3. Interfaccia Utente
# ==========================================

st.title("🌈 STENCIL PARTY! 🎨")
st.write("### Trasforma le tue foto in stencil pazzeschi in un lampo!")

# Sidebar
st.sidebar.header("⚙️ SETTAGGI SUPER")
num_layers = st.sidebar.slider("Quanti colori?", 2, 6, 3)
st.sidebar.markdown("---")
st.sidebar.subheader("🔗 PONTI MAGICI")
bridge_len = st.sidebar.slider("Lunghezza", 10, 100, 40)
bridge_thick = st.sidebar.slider("Spessore", 1, 15, 5)

uploaded_file = st.file_uploader("🚀 Carica la tua foto qui!", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Elaborazione
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("📸 Originale")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)

    # Elaborazione tecnica
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray, 9, 75, 75)
    data = smooth.reshape((-1, 1)).astype(np.float32)
    _, label, centers = cv2.kmeans(data, num_layers, None, (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(np.sort(centers.flatten()))
    quantized = centers[label.flatten()].reshape(smooth.shape)

    st.markdown("---")
    st.header("✂️ I TUOI LIVELLI DA RITAGLIARE")
    
    cols = st.columns(num_layers)
    for i in range(num_layers):
        layer_mask = cv2.inRange(quantized, int(centers[i]), int(centers[i]))
        final_layer = apply_bridges(layer_mask, bridge_len, bridge_thick)
        
        with cols[i]:
            st.markdown(f"**LIVELLO {i+1}**")
            st.image(final_layer, use_column_width=True)
            
            res, thumb = cv2.imencode(".png", final_layer)
            st.download_button(
                label=f"💥 SCARICA {i+1}",
                data=thumb.tobytes(),
                file_name=f"stencil_pop_{i+1}.png",
                mime="image/png"
            )

st.sidebar.success("💡 Consiglio: usa cartoncini colorati diversi per ogni strato!")

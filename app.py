import streamlit as st
import cv2
import numpy as np

# ==========================================
# 1. Configurazione e Stile (CSS)
# ==========================================
st.set_page_config(page_title="Stencil Van Gogh", layout="centered")

# CSS per sfondo Van Gogh sfocato e layout pulito
st.markdown("""
<style>
    /* Nascondi Sidebar */
    [data-testid="stSidebar"] { display: none; }

    /* Sfondo Van Gogh Sfumato */
    .stApp {
        background: radial-gradient(circle at 20% 20%, #1e3a5f 0%, #0d1a33 100%);
        background-attachment: fixed;
    }
    
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background-image: 
            radial-gradient(ellipse at 30% 30%, rgba(255, 230, 0, 0.1) 0%, transparent 40%),
            radial-gradient(ellipse at 70% 20%, rgba(200, 220, 255, 0.1) 0%, transparent 30%),
            linear-gradient(130deg, transparent 60%, rgba(10, 40, 90, 0.2) 70%, transparent 80%);
        filter: blur(15px);
        z-index: -1;
    }

    /* Testi e Titoli */
    h1, h2, h3 {
        color: #FFDE59 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        text-align: center;
    }

    /* Pulsante in fondo */
    .stButton>button {
        width: 100%;
        background-color: #FFDE59 !important;
        color: #0d1a33 !important;
        font-weight: bold !important;
        border-radius: 10px !important;
        padding: 20px !important;
        font-size: 1.2rem !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Funzioni Core
# ==========================================

def apply_bridges(mask, b_len, b_thick):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    output = mask.copy()
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, cnt in enumerate(contours):
            if hierarchy[i][3] != -1: # Se è un'isola
                if cv2.contourArea(cnt) > 80:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                        # Disegna il ponte bianco (interrompe il taglio nero)
                        cv2.line(output, (cX, cY), (cX, cY - b_len), 255, b_thick)
    return output

# ==========================================
# 3. Interfaccia Utente (Pagina Unica)
# ==========================================

st.title("🎨 Laboratorio Stencil Sognante")
st.write("Configura i tuoi parametri e genera gli strati per la bomboletta.")

# 1. Caricamento Immagine
st.header("1. Carica la foto")
uploaded_file = st.file_uploader("Scegli un file...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Mostra anteprima piccola
    st.image(uploaded_file, width=300)

    # 2. Impostazioni (Sotto l'immagine)
    st.header("2. Configurazione")
    col1, col2 = st.columns(2)
    with col1:
        num_layers = st.slider("Numero di strati", 2, 6, 3)
    with col2:
        bridge_thick = st.slider("Spessore ponti", 1, 10, 3)
    
    bridge_len = st.slider("Lunghezza connessioni", 10, 100, 40)

    st.markdown("---")

    # 3. Pulsante Genera (In Fondo)
    if st.button("✨ GENERA TUTTI I LIVELLI"):
        # Elaborazione
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        # Posterizzazione
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        smooth = cv2.bilateralFilter(gray, 9, 75, 75)
        data = smooth.reshape((-1, 1)).astype(np.float32)
        _, label, centers = cv2.kmeans(data, num_layers, None, 
                                      (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 
                                      10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(np.sort(centers.flatten()))
        quantized = centers[label.flatten()].reshape(smooth.shape)

        # Creazione Risultati
        st.header("🖼️ I tuoi strati pronti")
        for i in range(num_layers):
            mask = cv2.inRange(quantized, int(centers[i]), int(centers[i]))
            final_mask = apply_bridges(mask, bridge_len, bridge_thick)
            
            st.subheader(f"Livello {i+1} (Tonalità: {centers[i]})")
            st.image(final_mask, use_container_width=True)
            
            # Bottone Download
            res, thumb = cv2.imencode(".png", final_mask)
            st.download_button(
                label=f"📥 Scarica Livello {i+1}",
                data=thumb.tobytes(),
                file_name=f"stencil_layer_{i+1}.png",
                mime="image/png",
                key=f"dl_{i}"
            )

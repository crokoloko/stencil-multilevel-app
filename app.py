import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ==========================================
# 1. Configurazione e Stile (CSS)
# ==========================================
# Impostiamo il layout 'centered' per una migliore leggibilità su sfondo complesso
st.set_page_config(
    page_title="Il Laboratorio dello Stencil Sognante",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- CSS PERSONALIZZATO (Sfondo Van Gogh Sfocato) ---
vangogh_blur_css = """
<style>
    /* 1. Rimuovi Sidebar e Menu Streamlit */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* 2. Sfondo della Pagina (Lo Sfondo Van Gogh Sfocato) */
    .stApp {
        /* Creazione di un pattern di pennellate sfuocate (stile Starry Night sfocato) */
        background-color: #0d1a33; /* Blu scuro di base */
        background-image: 
            radial-gradient(ellipse at 30% 30%, rgba(255, 230, 0, 0.1) 0%, transparent 40%), /* Luna sfuocata */
            radial-gradient(ellipse at 70% 20%, rgba(200, 220, 255, 0.1) 0%, transparent 30%), /* Stelle sfuocate */
            
            /* Pennellate lunghe, sfuocate e vorticose */
            linear-gradient(130deg, transparent 60%, rgba(10, 40, 90, 0.2) 70%, transparent 80%),
            linear-gradient(220deg, transparent 30%, rgba(15, 60, 130, 0.2) 50%, transparent 70%),
            linear-gradient(40deg, transparent 10%, rgba(20, 80, 180, 0.2) 30%, transparent 50%),
            linear-gradient(310deg, transparent 0%, rgba(5, 20, 50, 0.2) 20%, transparent 40%),
            linear-gradient(170deg, transparent 40%, rgba(10, 30, 70, 0.2) 60%, transparent 80%);
        
        background-size: 150% 150%; /* Sovradimensionato per un effetto più morbido */
        
        /* Applichiamo la sfocatura (Blur) all'intero sfondo */
        filter: blur(8px);
        
        /* Assicuriamoci che i contenuti non siano sfuocati */
        transform: scale(1.05); /* Evita bordi neri per il blur */
        z-index: -1;
    }

    /* 3. Contenitore per i Contenuti (per contrasto) */
    .stMainBlock {
        background-color: rgba(0, 0, 0, 0.85); /* Sfondo scuro semitrasparente */
        color: #e0f0ff; /* Testo chiaro, bluastro */
        font-family: 'Garamond', 'Crimson Text', serif;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 2rem;
        margin-bottom: 2rem;
    }

    /* 4. Titoli e Intestazioni */
    @import url('https://fonts.googleapis.com/css2?family=UnifrakturMaguntia&display=swap');
    
    h1, h2, h3, .stHeader {
        font-family: 'UnifrakturMaguntia', cursive !important;
        color: #FFDE59 !important; /* Oro antico/Giallo Van Gogh */
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    h1 { font-size: 3rem !important; }

    /* 5. Pulsanti e Controlli */
    .stButton>button {
        background-color: rgba(255, 222, 89, 0.05) !important; /* Giallo sfuocato */
        color: #FFDE59 !important;
        border: 2px solid #FFDE59 !important;
        border-radius: 5px;
        font-family: 'Garamond', serif;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 0 10px rgba(255, 222, 89, 0.3);
        transition: all 0.3s;
        margin-top: 1rem;
        width: 100%; /* Pulsante a tutta larghezza */
    }
    .stButton>button:hover {
        background-color: #FFDE59 !important;
        box-shadow: 0 0 20px rgba(255, 222, 89, 0.7);
        color: #000000 !important;
    }
    
    /* Regola il testo nei controlli */
    [data-testid="stFileUploader"] label, .stSlider label {
        color: #e0f0ff !important;
    }
</style>
"""
st.markdown(vangogh_blur_css, unsafe_allow_html=True)

# ==========================================
# 2. Funzioni Logiche di Elaborazione
# ==========================================

def apply_bridges(mask, bridge_length, thickness):
    """Identifica le isole e crea i ponti di giunzione."""
    # Analisi dei contorni con gerarchia (RETR_CCOMP per i buchi)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    output = mask.copy()
    
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, cnt in enumerate(contours):
            # Se ha un padre (hierarchy[i][3] != -1), è un'isola/buco
            if hierarchy[i][3] != -1:
                area = cv2.contourArea(cnt)
                if area > 100: # Filtro dimensione minima (regolabile)
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        # Disegna il ponte verso l'alto (bianco su maschera nera)
                        cv2.line(output, (cX, cY), (cX, cY - bridge_length), 255, thickness)
    return output

def create_stencil_layers(image, num_layers, bridge_len, bridge_thick):
    """Genera gli strati stencil dall'immagine, applicando i ponti."""
    # Pre-processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray, 11, 85, 85) # Mantiene i bordi ma rimuove il rumore
    
    # K-Means Posterization
    data = smooth.reshape((-1, 1)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, centers = cv2.kmeans(data, num_layers, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Ordina i centri per luminosità (dal più scuro al più chiaro)
    centers = np.uint8(np.sort(centers.flatten()))
    quantized = centers[label.flatten()].reshape(smooth.shape)

    layers = []
    for i in range(num_layers):
        # 1. Isola il livello specifico (maschera binaria)
        # Il valore di grigio dello strato è centers[i]
        layer_mask = cv2.inRange(quantized, int(centers[i]), int(centers[i]))
        
        # 2. Applica i ponti automatici
        final_layer = apply_bridges(layer_mask, bridge_len, bridge_thick)
        layers.append((final_layer, centers[i]))
        
    return layers

# ==========================================
# 3. Interfaccia Utente (UI)
# ==========================================

st.title("Ars Stencil: Il Rituale Sognante")
st.write("🌌 Benvenuto, artefice. Carica il tuo simulacro e forgia gli strati del sogno.")

st.markdown("---")

# --- AREA DI CARICAMENTO ---
st.header("🕯️ Caricamento del Simulacro")
uploaded_file = st.file_uploader("Sacrifica un Simulacro (Immagine)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # --- Elaborazione Immagine (Corretto) ---
    # Conversione corretta dell'immagine per OpenCV
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    
    # --- Visualizzazione Originale ---
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Simulacro Pronto per il Rituale", use_column_width=True)

    st.markdown("---")

    # --- CONFIGURAZIONE DEGLI STENCIL ---
    st.header("📜 Rituali di Configurazione")
    st.write("Definisci la profondità e i legami dei tuoi strati.")

    num_layers_config = st.slider("🕯️ Numero di Livelli (Sogno)", 2, 8, 4)
    st.markdown("---")
    st.subheader("🔗 Ponti dell'Anima (Legami)")
    bridge_len_config = st.slider("Lunghezza Legami (px)", 10, 150, 40)
    bridge_thick_config = st.slider("Spessore Legami (px)", 1, 15, 4)

    st.markdown("---")

    # --- PULSANTE GENERA (In fondo alla pagina) ---
    generate_btn = st.button("✨ Genera Frammenti Stencil ✨")

    if generate_btn:
        # --- Generazione e Visualizzazione dei Livelli ---
        with st.spinner("⏳ Il rituale di posterizzazione è in corso..."):
            stencil_layers = create_stencil_layers(img_bgr, num_layers_config, bridge_len_config, bridge_thick_config)
        
        st.header("💀 I Frammenti del Sogno (Strati Stencil)")
        cols = st.columns(num_layers_config)
        
        for idx, (l_img, val) in enumerate(stencil_layers):
            # Visualizza e offri il download
            with cols[idx]:
                # Aggiungi un titolo sognante ad ogni colonna
                st.markdown(f"### Frammento {idx+1}")
                st.write(f"*(Grigio: {val})*")
                st.image(l_img, use_column_width=True)
                
                # Bottone di download stilizzato
                res, thumb = cv2.imencode(".png", l_img)
                st.download_button(
                    label=f"📥 Scarica Frammento {idx+1}",
                    data=thumb.tobytes(),
                    file_name=f"stencil_sognante_livello_{idx+1}.png",
                    mime="image/png",
                    key=f"btn_{idx}" # Aggiungiamo un key univoco per ogni bottone
                )

        st.markdown("---")
        st.warning("Memento: L'ordine di spruzzata va dal più scuro (Strato 1) al più chiaro.")

st.markdown("---")
st.info("💡 Consiglio: usa cartoncini di diverso colore per ogni strato!")

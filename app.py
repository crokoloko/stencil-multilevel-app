import streamlit as st
import cv2
import numpy as np
import base64
from PIL import Image

# ==========================================
# 1. Configurazione Iniziale e Stile Gotico (CSS)
# ==========================================
st.set_page_config(
    page_title="Il Laboratorio dello Stencil Gotico",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Funzione per caricare un'immagine locale come sfondo in base64 (opzionale)
# def get_base64_of_bin_file(bin_file):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

# --- CSS PERSONALIZZATO (Lo "Stile Gotico") ---
gothic_css = """
<style>
    /* 1. Sfondo e Testo Principale */
    .stApp {
        background-color: #1a0f1a; /* Viola molto scuro/Nero */
        color: #e0d0e0; /* Grigio perla antico */
        font-family: 'Garamond', 'Crimson Text', serif; /* Font eleganti e scuri */
    }

    /* 2. Titoli e Intestazioni (Carattere Gotico) */
    @import url('https://fonts.googleapis.com/css2?family=UnifrakturMaguntia&display=swap');
    
    h1, h2, h3, .stHeader {
        font-family: 'UnifrakturMaguntia', cursive !important; /* Il vero font Gotico */
        color: #8b0000 !important; /* Rosso scuro/sangue */
        text-shadow: 2px 2px 4px #000000;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    h1 { font-size: 3.5rem !important; text-align: center; }

    /* 3. Sidebar (Menu Laterale) */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #000000 !important;
        border-right: 2px solid #3d0a0a; /* Bordo rosso scuro */
    }
    
    [data-testid="stSidebar"] h2 {
        color: #c0c0c0 !important; /* Argento antico */
        font-size: 2rem;
    }

    /* 4. Pulsanti e Controlli */
    .stButton>button {
        background-color: #3d0a0a !important; /* Rosso scuro quasi nero */
        color: #e0d0e0 !important;
        border: 2px solid #8b0000 !important;
        border-radius: 5px;
        font-family: 'Garamond', serif;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 0 10px rgba(139, 0, 0, 0.5);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #8b0000 !important;
        box-shadow: 0 0 20px rgba(224, 208, 224, 0.7);
        color: #000000 !important;
    }

    /* 5. Caricamento File e Dropzone */
    [data-testid="stFileUploader"] {
        border: 3px dashed #3d0a0a !important;
        border-radius: 10px;
        background-color: #120912;
    }
    [data-testid="stFileUploader"] label {
        font-family: 'UnifrakturMaguntia', cursive;
        color: #8b0000 !important;
        font-size: 1.5rem !important;
    }

    /* 6. Tabelle e Messaggi d'Errore (se ce ne sono) */
    .stAlert {
        background-color: #3d0a0a;
        color: #e0d0e0;
        border: 2px solid #8b0000;
    }
</style>
"""
st.markdown(gothic_css, unsafe_allow_html=True)

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
                        # Nota: Questa è una logica di base "verso l'alto".
                        cv2.line(output, (cX, cY), (cX, cY - bridge_length), 255, thickness)
    return output

# ==========================================
# 3. Interfaccia Utente (UI) Gotica
# ==========================================

st.title("Ars Stencil: Il Laboratorio Oscuro")
st.write("🌌 Benvenuto, artefice. Carica il tuo simulacro e forgia gli strati dell'oscurità per i tuoi stencil.")

# --- Sidebar per i Parametri ---
st.sidebar.header("📜 Rituali di Configurazione")
st.sidebar.write("Definisci la profondità e i legami dei tuoi strati.")

num_layers = st.sidebar.slider("🕯️ Numero di Livelli (Oscurità)", 2, 8, 4)
st.sidebar.markdown("---")
st.sidebar.subheader("🔗 Ponti dell'Anima (Giunzioni)")
bridge_len = st.sidebar.slider("Lunghezza Legami (px)", 10, 150, 40)
bridge_thick = st.sidebar.slider("Spessore Legami (px)", 1, 15, 4)

# --- Area di Caricamento ---
uploaded_file = st.file_uploader("🕯️ Sacrifica un Simulacro (Immagine)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # --- Elaborazione Immagine (Corretto) ---
    # Conversione corretta dell'immagine per OpenCV
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Pre-processing Gotico (Bilateral Filter per mantenere i bordi puliti)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray, 11, 85, 85)
    
    # --- Visualizzazione Originale ---
    col_orig, col_process = st.columns(2)
    with col_orig:
        st.header("🔮 Il Simulacro Originale")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)

    # --- K-Means Posterization (Creazione Strati) ---
    st.sidebar.info("⏳ Il rituale di posterizzazione è in corso...")
    data = smooth.reshape((-1, 1)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, centers = cv2.kmeans(data, num_layers, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Ordina i centri per luminosità (dal più scuro al più chiaro)
    centers = np.uint8(np.sort(centers.flatten()))
    quantized = centers[label.flatten()].reshape(smooth.shape)

    # --- Generazione e Visualizzazione dei Livelli ---
    st.header("💀 I Frammenti dell'Oscurità (Strati Stencil)")
    cols = st.columns(num_layers)
    
    for i in range(num_layers):
        # 1. Isola il livello specifico (maschera binaria)
        # Il valore di grigio dello strato è centers[i]
        layer_mask = cv2.inRange(quantized, int(centers[i]), int(centers[i]))
        
        # 2. Applica i ponti automatici
        final_layer = apply_bridges(layer_mask, bridge_len, bridge_thick)
        
        # 3. Visualizza e offri il download
        with cols[i]:
            # Aggiungi un titolo gotico ad ogni colonna
            st.markdown(f"### Strato {i+1}")
            st.write(f"*(Grigio: {centers[i]})*")
            st.image(final_layer, use_column_width=True)
            
            # Bottone di download stilizzato
            res, thumb = cv2.imencode(".png", final_layer)
            st.download_button(
                label=f"📥 Scarica Frammento {i+1}",
                data=thumb.tobytes(),
                file_name=f"stencil_oscuro_livello_{i+1}.png",
                mime="image/png"
            )

st.sidebar.markdown("---")
st.sidebar.warning("Memento: L'ordine di spruzzata va dal più scuro (Strato 1) al più chiaro.")

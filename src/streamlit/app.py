import streamlit as st
import pathlib
import sys
import io
import cv2
import numpy as np

# To load our custom model functions
path = pathlib.Path("../models").resolve()
sys.path.insert(0, str(path))
path = pathlib.Path("..").resolve()
sys.path.insert(0, str(path))

# clean for reloading scriptwithout spamming sys.path insert
sys.path = list(dict.fromkeys(sys.path))

from model_functions import load_model, keras_predict_model, get_predict_value
from data_functions import load_resize_img_from_buffer
from predict_model import get_predictions

# App config:
model_save_path="../../models/EfficientNetB4_masked-Covid-19_masked-91.45.keras"
seg_model_save_path="../../models/unet-0.98.keras"


# import custom Navigation bar
from modules.nav import Navbar
Navbar()

st.title("Application de classification de Radiographie Pulmonaire")

context_text = """
<div style="text-align: justify;">

Cette aplication permet la prédiction de l'état d'un patient à partir image de radiographie pulmonaire.

La prédiction est effectué par le model de deep-learning **EfficientNetB4**. Ce model est entrainé pour classifier l'image parmis les 4 possibilité suivantes: "sain", "atteint du  Covid", "de pneumonie" ou "d'opacité pulmonaire".

Le fichier à uploader peut être une image au format "png", "jpg", ou un dossier au format "zip" pour prédire un ensemble directement.

Des exemples sont fournis ci-dessous pour pouvoir tester l'application.

</div>"""
st.markdown(context_text, unsafe_allow_html=True)

uploaded_file = st.file_uploader("Fichier ou dossier à prédire:", type=['png', 'jpg', 'zip'])
if uploaded_file is not None:
    f_type = uploaded_file.type.split("/")[-1]
    if f_type in ['png', 'jpg']:
        # si png, jpg

        img = load_resize_img_from_buffer(uploaded_file)
        st.image(img, caption="Image loaded after re-sizing:", use_container_width=False)

        masked_value = st.selectbox("Est-ce que l'image est masqué ? (Ne présente que les poumons et pas le coeur, foie et autre marquage)*",
                                    ("Oui", "Non"))
        masked_value = True if masked_value == "Oui" else False

        if st.button("Démarrer la prediction", icon="🚀"):
            # load model
            model = load_model(model_save_path)

            # make predict
            pred = keras_predict_model(model, img)

            # Interpret prediction
            pred = get_predict_value(pred)

            st.text(pred)

        # make gradcam

    elif f_type == "zip":
        # si zip prediction sur folder, retour que d'un df avec name / prediction
        print("A folder to process")


    else:
        raise TypeError("Wrong file type submitted (not 'png', 'jpg' or 'zip')")
        


import streamlit as st
import pathlib
import sys

# To load our custom model functions
#path = pathlib.Path("..").resolve()
path = pathlib.Path("../models").resolve()
sys.path.insert(0, str(path))

from models.predict_model import get_predictions

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

uploaded_file = st.file_uploader("Fichier à prédire:", type=['png', 'jpg','zip'])
if uploaded_file is not None:
    st.text(uploaded_file)
    print(uploaded_file.name)

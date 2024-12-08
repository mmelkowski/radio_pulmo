import streamlit as st
import pathlib
import sys
import random
from io import BytesIO

# To load our custom model functions
path = pathlib.Path("../models").resolve()
sys.path.insert(0, str(path))
path = pathlib.Path("..").resolve()
sys.path.insert(0, str(path))

## clean for reloading scriptwithout spamming sys.path insert
sys.path = list(dict.fromkeys(sys.path))

from data_functions import load_resize_img_from_buffer

# import custom streamlit script
from modules.nav import Navbar
from modules.img_functions import convert_array_to_PIL, convert_PIL_to_io
from modules.actions_functions import (
    action_prediction,
    action_visualization,
    action_masking,
)


# App config:
model_save_path = "../../models/EfficientNetB4_masked-Covid-19_masked-91.45.keras"
seg_model_save_path = "../../models/cxr_reg_segmentation.best.keras"

# Streamlit app page
Navbar()

st.title("Application de classification de Radiographie Pulmonaire")

help_tooltip = """La prédiction est effectuée par le modèle de deep-learning **EfficientNetB4**. 
Ce modèle est entrainé pour classifier une radiographie pulmonaire parmi les 4 possibilités suivantes: "sain", "atteint du  Covid", "de pneumonie virale" ou "d'opacité pulmonaire".
Sa précision est de 92% pour l'ensemble des catégories.

Plus d'informations sont disponibles dans les parties "Contexte" et "Modélisation".
"""

context_text = """
<div style="text-align: justify;">

Cette application permet la prédiction de l'état d'un patient à partir d'une radiographie pulmonaire pour les affections suivantes : Covid, pneumonie virale ou opacité pulmonaire.
</div>"""
st.markdown(context_text, unsafe_allow_html=True, help=help_tooltip)

text1 = """
<div style="text-align: justify;">
La prédiction s'effectue sur des images brutes ou après isolation du poumon.
<br> Il est possible d'effectuer la prédiction et de visualiser les zones les plus informatives.
<br>
<br>
Elle peut être utilisée à partir des exemples fournis ci-dessous ou en important vos propres images.
</div>"""
st.markdown(text1, unsafe_allow_html=True)


# Répertoire contenant les fichiers d'exemple
ex_dir =  pathlib.Path("resources/ex_images")

# Liste des fichiers d'exemple dans le répertoire
example_files = ["Aucun"] + [f.name for f in ex_dir.iterdir() if f.is_file() and f.suffix in ['.png', '.jpg', '.jpeg', '.zip']]

# Sélectionner un fichier d'exemple via un selectbox
selected_file = st.selectbox("Choisir un fichier d'exemple", example_files)


context_text_2 = """
<div style="text-align: justify;">
Le fichier à importer doit être une image au format "png", "jpg".
<br> Il est possible de prédire un ensemble de fichiers avec un dossier au format "zip".

</div>"""
st.markdown(context_text_2, unsafe_allow_html=True)


uploaded_file = st.file_uploader(
    "Fichier ou dossier à prédire:", type=["png", "jpg", "jpeg", "zip"]
)
if uploaded_file is not None:
    f_type = uploaded_file.type.split("/")[-1]
    filename = uploaded_file.name

# Si un fichier d'exemple est sélectionné, on l'ouvre directement
if selected_file:
    if selected_file != "Aucun" :
        file_path =  pathlib.Path("resources/ex_images",selected_file)
        with open(file_path, "rb") as f:
            file_content = f.read()
        uploaded_file = BytesIO(file_content)
        f_type = selected_file.split(".")[-1]
        filename = selected_file

if uploaded_file is not None or selected_file != 'Aucun':
    # ne fonctionne plus avec fichier exemple on le passe au dessus
    #f_type = uploaded_file.type.split("/")[-1]
    if f_type in ["png", "jpg", "jpeg"] :

        # si png, jpg

        img_original_array, img = load_resize_img_from_buffer(
            uploaded_file, target_size=(224, 224)
        )

        st.image(
            img,
            caption="Image chargée après redimensionnement",
            use_container_width=False,
        )

        action_required = st.selectbox(
            "Voulez vous prédire, masquer ou visualiser (*Grad-CAM*) l'image ?",
            ("Prédire", "Masquer", "Visualiser"),
        )

        if action_required == "Prédire":
            help_masked_value = "Si 'Non' alors le modèle de segmentation procèdera au masquage automatiquement avant la prédiction."
            masked_value = st.selectbox(
                "L'image est-elle masquée ? (*Les poumons sont isolés, on ne voit ni l'arrière-plan ni les autres organes*)",
                ("Oui", "Non"),
                help=help_masked_value,
            )
            masked_value = True if masked_value == "Oui" else False

            left, middle, right = st.columns(3)
            if middle.button("Démarrer la prédiction", icon="🚀"):
                with st.status("Prédiction en cours...", expanded=True):
                    pred = action_prediction(
                        model_save_path,
                        img,
                        masked_value=masked_value,
                        seg_model_save_path="../../models/cxr_reg_segmentation.best.keras",
                    )

                st.success("Prédiction effectuée")

                st.text(f" L'image est classée comme: {pred}")

        elif action_required == "Visualiser":

            
            vis_text = """
            <div style="text-align: justify;">
            Pour la visualisation, l'image doit être masquée.
            <br> Vous pouvez visualiser les pixels les plus importantes au début, au milieu et à la fin de prédiction.
            </div>"""
            st.markdown(vis_text, unsafe_allow_html=True)

            layer_name = st.selectbox(
                "Choix de la couche à visualiser (Première : stem_conv, Intermédiaire : block4f_expand_conv, Finale : top_conv):",
                ("stem_conv", "block4f_expand_conv", "top_conv"),
                index=2
            )

            left, middle, right = st.columns(3)
            if middle.button("Démarrer la visualisation", icon="🔍"):

                with st.status("Visualisation en cours...", expanded=True):
                    heatmap, overlay = action_visualization(
                        model_save_path, img, img_original_array, layer_name
                    )

                st.success("Visualisation effectuée")

                # Display image
                heatmap_PIL = convert_array_to_PIL(heatmap)
                overlay_PIL = convert_array_to_PIL(overlay)
                left_img, right_img = st.columns(2)
                left_img.image(
                    overlay_PIL, caption="Grad-CAM Applied", use_container_width=False
                )
                right_img.image(
                    heatmap_PIL, caption="Heatmap generated", use_container_width=False
                )

                # Convert for buffering
                left_d, right_d = st.columns(2)
                io_heatmap = convert_PIL_to_io(heatmap_PIL, img_format="PNG")
                io_overlay = convert_PIL_to_io(overlay_PIL, img_format="PNG")

                # Download
                left_d.download_button(
                    label="Download Grad-CAM",
                    data=io_overlay,
                    file_name=f"Grad_CAM_{filename}.png",
                    mime="image/png",
                )
                right_d.download_button(
                    label="Download heatmap",
                    data=io_heatmap,
                    file_name=f"Heatmap_{filename}.png",
                    mime="image/png",
                )

        elif action_required == "Masquer":
            if st.button("Démarrer le masquage", icon="👺"):
                with st.status("Masquage en cours...", expanded=True):
                    mask, masked_img = action_masking(
                        seg_model_save_path, img_original_array
                    )

                st.success("Masquage effectué")

                left_img, right_img = st.columns(2)
                left_img.image(
                    masked_img, caption="Image masquée", use_container_width=False
                )
                right_img.image(
                    mask, caption="Masque", use_container_width=False, clamp=True
                )

    elif f_type == "zip":
        # si zip prediction sur folder, retour que d'un df avec name / prediction
        print("A folder to process")

    else:
        print("uploaded_file.type", uploaded_file.type)
        raise TypeError("Wrong file type submitted (not 'png', 'jpg', 'jpeg' or 'zip')")

# bottom_text = """
# <div style="font-size: 12px; color: gray; font-style: italic; position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%); margin: 0;">

# Cette application a été développée par [Chris Hozé](https://www.linkedin.com/in/chris-hozé-007901a5) et [Mickaël Melkowski](https://www.linkedin.com/in/mickael-melkowski/).
# </div>
# """
# st.markdown(bottom_text, unsafe_allow_html=True)



# Crédit
bottom_text = """
<div style="font-size: 14px; color: gray; font-style: italic; text-align: center; margin-top: 20px;">
 Cette application a été développée par 
    <br>
    <a href="https://www.linkedin.com/in/chris-hozé-007901a5" target="_blank" style="color: #0073e6;">Chris Hozé</a> 
    et 
    <a href="https://www.linkedin.com/in/mickael-melkowski/" target="_blank" style="color: #0073e6;">Mickaël Melkowski</a>
     <br> dans le cadre de notre formation en DataScience réalisée avec DataScientest.

</div>
"""

# Affichage du texte dans la sidebar
st.sidebar.markdown(bottom_text, unsafe_allow_html=True)

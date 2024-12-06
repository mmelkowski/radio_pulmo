import streamlit as st
import pathlib
import sys

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
Ce modèle est entrainé pour classifier l'image parmi les 4 possibilités suivantes: "sain", "atteint du  Covid", "de pneumonie virale" ou "d'opacité pulmonaire".

Plus d'informations sont disponibles dans la partie "Contexte" et "Modélisation".
"""

context_text = """
<div style="text-align: justify;">

Cette application permet la prédiction de l'état d'un patient à partir d'une radiographie pulmonaire.

</div>"""
st.markdown(context_text, unsafe_allow_html=True, help=help_tooltip)


context_text_2 = """
<div style="text-align: justify;">
Le fichier à importer peut-être une image au format "png", "jpg", ou un dossier au format "zip" pour prédire un ensemble directement.

Des exemples sont fournis ci-dessous pour pouvoir tester l'application.

</div>"""
st.markdown(context_text_2, unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Fichier ou dossier à prédire:", type=["png", "jpg", "jpeg", "zip"]
)
if uploaded_file is not None:
    f_type = uploaded_file.type.split("/")[-1]
    if f_type in ["png", "jpg", "jpeg"]:
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
<<<<<<< HEAD
            help_masked_value = "Si 'Non' alors le modèle de segmentation procèdera au masquage avant la prédiction."
=======
            help_masked_value = "Si 'Non' alors le model de segmentation procèdera au masquage automatiquement avant la prédiction."
>>>>>>> 0540b53 (UPDATE(streamlit): Remove Unused import.)
            masked_value = st.selectbox(
                "Est-ce que l'image est masquée ? (*Les poumons sont isolés, on ne voit pas l'arrière-plan et les autres organes*)",
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
                    file_name=f"Grad_CAM_{uploaded_file.name}.png",
                    mime="image/png",
                )
                right_d.download_button(
                    label="Download heatmap",
                    data=io_heatmap,
                    file_name=f"Heatmap_{uploaded_file.name}.png",
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

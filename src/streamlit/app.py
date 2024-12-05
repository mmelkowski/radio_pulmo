import streamlit as st
import pathlib
import sys
import numpy as np
import matplotlib.cm as cm
from PIL import Image
import io
import cv2

# To load our custom model functions
path = pathlib.Path("../models").resolve()
sys.path.insert(0, str(path))
path = pathlib.Path("..").resolve()
sys.path.insert(0, str(path))

## clean for reloading scriptwithout spamming sys.path insert
sys.path = list(dict.fromkeys(sys.path))

from model_functions import load_model, keras_predict_model, get_predict_value
from seg_predict_model import scale_image, mask_generation, apply_mask
from data_functions import load_resize_img_from_buffer
from plot_functions import make_gradcam_heatmap, overlay_heatmap_on_array

# import custom streamlit script
from modules.nav import Navbar
from modules.img_functions import convert_array_to_PIL, convert_PIL_to_io


# App config:
model_save_path="../../models/EfficientNetB4_masked-Covid-19_masked-91.45.keras"
seg_model_save_path="../../models/unet-0.98.keras"
cmap = cm.viridis

Navbar()

st.title("Application de classification de Radiographie Pulmonaire")

help_tooltip = """La prédiction est effectué par le model de deep-learning **EfficientNetB4**. 
Ce model est entrainé pour classifier l'image parmis les 4 possibilité suivantes: "sain", "atteint du  Covid", "de pneumonie" ou "d'opacité pulmonaire".

Plus d'information dans la partie "Contexte" et "modélisation".
"""

context_text = """
<div style="text-align: justify;">

Cette aplication permet la prédiction de l'état d'un patient à partir image de radiographie pulmonaire.

</div>"""
st.markdown(context_text, unsafe_allow_html=True, help=help_tooltip)


context_text_2 = """
<div style="text-align: justify;">
Le fichier à uploader peut être une image au format "png", "jpg", ou un dossier au format "zip" pour prédire un ensemble directement.

Des exemples sont fournis ci-dessous pour pouvoir tester l'application.

</div>"""
st.markdown(context_text_2, unsafe_allow_html=True)

uploaded_file = st.file_uploader("Fichier ou dossier à prédire:", type=['png', 'jpg', 'jpeg', 'zip'])
if uploaded_file is not None:
    f_type = uploaded_file.type.split("/")[-1]
    if f_type in ['png', 'jpg', 'jpeg']:
        # si png, jpg

        img_original_array, img = load_resize_img_from_buffer(uploaded_file, target_size=(224,224))

        st.image(img, caption="Image loaded after re-sizing", use_container_width=False)

        action_required = st.selectbox("Voulez vous prédire, masquer ou visualiser (Grad-CAM) l'image ?",
                                       ("Prédire", "Masquer", "Visualiser"))

        if action_required == "Prédire":
            masked_value = st.selectbox("Est-ce que l'image est masquée ? (Ne présente que les poumons et pas le coeur, foie et autre marquage)*",
                                        ("Oui", "Non"))
            masked_value = True if masked_value == "Oui" else False

            left, middle, right = st.columns(3)
            if middle.button("Démarrer la prediction", icon="🚀"):
                with st.status("Prediction en cours...", expanded=True):

                    if not masked_value:
                        st.write("Masquage à faire...")
                        st.write("Chargement du model de segmentation...")
                        # load model
                        seg_model = load_model(seg_model_save_path)


                    st.write("Chargement du model de prediction...")
                    # load model
                    model = load_model(model_save_path)

                    # make predict
                    st.write("Prédiction...")
                    pred = keras_predict_model(model, img)

                    # Interpret prediction
                    st.write("Interpretation...")
                    pred = get_predict_value(pred)

                    st.write("Prédiction finie.")

                st.success("Prédiction effectuée")

                st.text(f" L'image est classée comme: {pred}")

        elif action_required == "Visualiser":

            layer_name = st.selectbox("Choix de la layer à visualiser (Trié dans l'ordre: première, milieu, dernière):",
                                      ("stem_conv", "block4f_expand_conv", "top_conv"))

            left, middle, right = st.columns(3)
            if middle.button("Démarrer la visualisation", icon="🚀"):

                with st.status("Visualisation en cours...", expanded=True):

                    st.write("Chargement du model...")
                    # Load model
                    model = load_model(model_save_path)
                    efficientnet = model.get_layer("efficientnetb4")

                    st.write("Création de l'overlay...")
                    # Make gradcam
                    heatmap = make_gradcam_heatmap(img, efficientnet, layer_name)

                    # Overlay the heatmap on the original image
                    overlay = overlay_heatmap_on_array(heatmap, img_original_array, alpha=0.4)

                    st.write("Grad-CAM fini.")

                # Display image
                heatmap_PIL = convert_array_to_PIL(heatmap)
                overlay_PIL = convert_array_to_PIL(overlay)
                left_img, right_img = st.columns(2)
                left_img.image(overlay_PIL, caption="Grad-CAM Applied", use_container_width=False)
                right_img.image(heatmap_PIL, caption="Heatmap generated", use_container_width=False)

                # Download
                left_d, right_d = st.columns(2)
                io_heatmap = convert_PIL_to_io(heatmap_PIL, img_format="PNG")
                io_overlay = convert_PIL_to_io(overlay_PIL, img_format="PNG")

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
            if st.button("Démarrer le masquage", icon="🚀"):
                with st.status("Masquage en cours...", expanded=True):
                    # load model 
                    st.write("Chargement du model...")
                    model_seg = load_model(seg_model_save_path)

                    # masking
                    st.write("Masquage...")

                    img_to_mask = cv2.resize(img_original_array, dsize=(256, 256))
                    img_to_mask = scale_image(img_to_mask)
                    img_to_mask = np.array(img_to_mask).reshape(1, 256, 256, 1)
                    mask = mask_generation(model_seg, img_to_mask)
                    mask = np.array(mask).reshape(256, 256)
                    masked_img = apply_mask(img_original_array, mask, resize=True, width=img_original_array.shape[0], height=img_original_array.shape[1])

                left_img, right_img = st.columns(2)
                left_img.image(masked_img, caption="masked_img", use_container_width=False)
                right_img.image(mask, caption="mask", use_container_width=False, clamp=True)

    elif f_type == "zip":
        # si zip prediction sur folder, retour que d'un df avec name / prediction
        print("A folder to process")

    else:
        print("uploaded_file.type", uploaded_file.type)
        raise TypeError("Wrong file type submitted (not 'png', 'jpg', 'jpeg' or 'zip')")
        


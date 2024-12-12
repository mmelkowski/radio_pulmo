import streamlit as st
import os
import pathlib
import shutil
from io import BytesIO

# import custom streamlit script
from modules.nav import Navbar
from modules.data_functions import load_resize_img_from_buffer
from modules.img_functions import convert_array_to_PIL, convert_PIL_to_io, read_zip_file
from modules.actions_functions import (
    action_prediction,
    action_visualization,
    action_masking,
    unzip_images,
    folder_action_prediction,
    folder_action_masking,
    folder_action_visualization
)
from modules.merge_model import merge_files
from modules.model_functions import load_model


@st.cache_resource(show_spinner="⏳ Chargement des modèles de Deep Learning...")
def load_and_cache_model(model_path):
    """Wrapper function to cache the loaded model

    Args:
        model_path: Path to the keras model saved
    """
    return load_model(model_path)

# page settings
st.set_option("client.showSidebarNavigation", False)
st.set_page_config(page_title="Radio-Pulmo Prediction App", page_icon="src/streamlit/resources/x-ray.ico")

# App config:
model_save_path = "models/EfficientNetB4_masked-Covid-19_masked-91.45.keras"
seg_model_save_path = "models/cxr_reg_segmentation.best.keras"
path_to_resources = pathlib.Path("src/streamlit/resources")

# Due to file size limitation the model is stored as seperate files to be merged when the app is running
if not pathlib.Path(model_save_path).exists():
    parts = [
        "models/EfficientNetB4_masked-Covid-19_masked-91.45.keras.part_0",
        "models/EfficientNetB4_masked-Covid-19_masked-91.45.keras.part_1",
        "models/EfficientNetB4_masked-Covid-19_masked-91.45.keras.part_2"
    ]
    output_file_path = "models/EfficientNetB4_masked-Covid-19_masked-91.45.keras"
    merge_files(output_file_path, parts)

#unzip temp folder path:
zip_folder_tmp = pathlib.Path(".temp")
zip_folder_tmp_raw = zip_folder_tmp / "raw"
zip_folder_tmp_processed = zip_folder_tmp / "processed"
zip_folder_tmp.mkdir(parents=True, exist_ok=True)
zip_folder_tmp_raw.mkdir(parents=True, exist_ok=True)
zip_folder_tmp_processed.mkdir(parents=True, exist_ok=True)

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

# <br> Elle peut être utilisée à partir des exemples fournis ci-dessous ou en important vos propres images.

st.markdown(context_text, unsafe_allow_html=True, help=help_tooltip)

text_1 = """
<div style="text-align: justify;">
La prédiction s'effectue sur des images brutes ou après isolation du poumon par masquage.
<br> Trois fonctionnalités sont disponibles :

- La prédiction des affections
- La génération de masque pour isoler les poumons
- La visualisation des zones les plus informatives pour la prédiction

</div>
"""

st.markdown(text_1, unsafe_allow_html=True)

# Répertoire contenant les fichiers d'exemple
ex_dir =  path_to_resources / "ex_images"

# Liste des fichiers d'exemple dans le répertoire
example_files = ["Aucun"] + [f.name for f in ex_dir.iterdir() if f.is_file() and f.suffix in ['.png', '.jpg', '.jpeg', '.zip']]

context_text_2 = """
<div style="text-align: justify;">

Le fichier à importer doit être une image au format "png", "jpg".
<br> Il est possible de prédire un ensemble de fichiers avec un dossier au format "zip".
<br>
</div>
"""

st.markdown(context_text_2, unsafe_allow_html=True)

#########################
# load model before action selection but after first text element to let the user wait
model = load_and_cache_model(model_save_path)
seg_model = load_and_cache_model(seg_model_save_path)
#########################

uploaded_file = st.file_uploader(
    "Fichier ou dossier à prédire:", type=["png", "jpg", "jpeg", "zip", "x-zip-compressed"]
)

# Sélectionner un fichier d'exemple via un selectbox
selected_file = st.selectbox("Sinon choisir un fichier d'exemple", example_files)

if uploaded_file is not None:
    f_type = uploaded_file.type.split("/")[-1]
    filename = uploaded_file.name

# Si un fichier d'exemple est sélectionné, on l'ouvre directement
if selected_file:
    if selected_file != "Aucun" :
        file_path =  path_to_resources / "ex_images" / selected_file
        with open(file_path, "rb") as f:
            file_content = f.read()
        uploaded_file = BytesIO(file_content)
        f_type = selected_file.split(".")[-1]
        filename = selected_file

if uploaded_file is not None or selected_file != 'Aucun':
    if f_type in ["png", "jpg", "jpeg"] :

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
                        model,
                        img,
                        masked_value=masked_value,
                        seg_model=seg_model,
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
                        model, img, img_original_array, layer_name
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
                    label="Telecharger l'image Grad-CAM",
                    data=io_overlay,
                    file_name=f"Grad_CAM_{filename}.png",
                    mime="image/png",
                )
                right_d.download_button(
                    label="Telecharger la Heatmap",
                    data=io_heatmap,
                    file_name=f"Heatmap_{filename}.png",
                    mime="image/png",
                )

        elif action_required == "Masquer":
            left, middle, right = st.columns(3)
            if middle.button("Démarrer le masquage", icon="👺"):
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

    elif f_type in ["zip", "x-zip-compressed"]:

        action_required = st.selectbox(
            "Voulez vous prédire, masquer ou visualiser (*Grad-CAM*) les images ?",
            ("Prédire", "Masquer", "Visualiser"),
        )

        if action_required == "Prédire":
            help_masked_value = "Si 'Non' alors le modèle de segmentation procèdera au masquage automatiquement avant la prédiction."
            masked_value = st.selectbox(
                "Est-ce que les images sont masquées ? (*Les poumons sont isolés, on ne voit pas l'arrière-plan et les autres organes*)",
                ("Oui", "Non"),
                help=help_masked_value,
            )
            left, middle, right = st.columns(3)
            if middle.button("Démarrer la prédiction", icon="🚀"):
                with st.status("Prédiction en cours...", expanded=True):
                    unzip_images(uploaded_file, extract_path=zip_folder_tmp_raw)

                    # make process folder
                    pathlib.Path(zip_folder_tmp_processed).mkdir(parents=True, exist_ok=True)

                    df, output = folder_action_prediction(
                        model,
                        zip_folder_tmp_raw,
                        masked_value=masked_value == "Oui",
                        seg_model=seg_model,
                    )
                
                st.success("Prédictions effectuées")

                st.write("Le tableau montre pour chaque fichier son résultat le plus probable et ensuite la probabilité par catégorie.")
                st.write(df)

                shutil.rmtree(zip_folder_tmp)

                st.download_button(
                    label="Télécharger les résultats en csv 📝",
                    data=output,
                    file_name=f"prediction_results.csv",
                    mime="text/csv",
                )

                del df
                del output

        elif action_required == "Masquer":
            left, middle, right = st.columns(3)
            if middle.button("Démarrer le masquage", icon="👺"):
                with st.status("Masquage en cours...", expanded=True):
                    #unzipping images
                    unzip_images(uploaded_file, extract_path=zip_folder_tmp_raw)

                    # make process folder
                    pathlib.Path(zip_folder_tmp_processed).mkdir(parents=True, exist_ok=True)

                    folder_action_masking(
                        seg_model, 
                        zip_folder_tmp_raw,
                        zip_folder_tmp_processed
                    )

                # zip zip_folder_tmp_raw
                shutil.make_archive("masked_images", 'zip', root_dir=zip_folder_tmp_processed)

                shutil.rmtree(zip_folder_tmp_processed)

                st.success("Masquage effectué")

                zip_data = read_zip_file("masked_images.zip")

                left, middle, right = st.columns(3)
                middle.download_button(
                    label="Télécharger les images masquées au format '.zip' 📦",
                    data=zip_data,
                    file_name="masked_images.zip",
                    mime="application/zip",
                )
                os.remove("masked_images.zip")
                del zip_data


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
                    #unzipping images
                    unzip_images(uploaded_file, extract_path=zip_folder_tmp_raw)

                    # make process folder
                    pathlib.Path(zip_folder_tmp_processed).mkdir(parents=True, exist_ok=True)

                    folder_action_visualization(
                        model,
                        layer_name,
                        zip_folder_tmp_raw,
                        zip_folder_tmp_processed
                    )

                st.success("Visualisation effectuée")

                # zip zip_folder_tmp_raw
                shutil.make_archive("images", 'zip', root_dir=zip_folder_tmp_processed)

                shutil.rmtree(zip_folder_tmp_processed)

                zip_data = read_zip_file("images.zip")

                left, middle, right = st.columns(3)
                middle.download_button(
                    label="Télécharger les visualisations au format '.zip' 📦",
                    data=zip_data,
                    file_name="images.zip",
                    mime="application/zip",
                )
                os.remove("images.zip")
                del zip_data

    else:
        print("uploaded_file.type", uploaded_file.type)
        raise TypeError("Wrong file type submitted (not 'png', 'jpg', 'jpeg' or 'zip')")

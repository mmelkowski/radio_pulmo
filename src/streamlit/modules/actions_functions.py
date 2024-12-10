import streamlit as st
import os
import numpy as np
import pathlib
import sys
import cv2
import zipfile
import pandas as pd

a="""
# To load our custom model functions
path = pathlib.Path("../../models").resolve()
sys.path.insert(0, str(path))
path = pathlib.Path("../..").resolve()
sys.path.insert(0, str(path))

## clean for reloading scriptwithout spamming sys.path insert
sys.path = list(dict.fromkeys(sys.path))
"""

from modules.model_functions import load_model, keras_predict_model, get_predict_value
from modules.seg_predict_model import scale_image, mask_generation, apply_mask
from modules.plot_functions import make_gradcam_heatmap, overlay_heatmap_on_array
from modules.data_functions import load_file


def action_visualization(model_save_path, img, img_original_array, layer_name):
    """Visualizes the model's decision-making process using Grad-CAM.

    This function loads a pre-trained model, generates a Grad-CAM heatmap for a given input image,
    and overlays the heatmap on the original image.

    Args:
        model_save_path: Path to the saved model.
        img: The input image as a NumPy array.
        img_original_array: The original image as a NumPy array.
        layer_name: The name of the layer in the model for which to generate Grad-CAM.

    Returns:
    A tuple containing:
        - The generated heatmap as a NumPy array.
        - The original image with the overlaid heatmap as a NumPy array.
    """
    # Load model
    st.write("⏳ Chargement du model de prédiction...")
    model = load_model(model_save_path)
    efficientnet = model.get_layer("efficientnetb4")

    st.write("🖌️ Création de l'overlay...")
    # Make gradcam
    heatmap = make_gradcam_heatmap(img, efficientnet, layer_name)

    # Overlay the heatmap on the original image
    overlay = overlay_heatmap_on_array(heatmap, img_original_array, alpha=0.4)

    return heatmap, overlay


def action_masking(seg_model_save_path, img_original_array):
    """Applies a segmentation mask to an image.

    This function loads a segmentation model, resizes the input image, generates a segmentation
    mask, and applies the mask to the original image.

    Args:
        seg_model_save_path: Path to the saved segmentation model.
        img_original_array: The original image as a NumPy array.

    Returns:
        A tuple containing:
        - The generated mask as a NumPy array.
        - The masked image as a NumPy array.
    """
    # load model
    st.write("⏳ Chargement du model de masquage...")
    model_seg = load_model(seg_model_save_path)

    # masking
    st.write("✏️ Calcul du Masque...")

    img_to_mask = cv2.resize(img_original_array, dsize=(256, 256))
    img_to_mask = scale_image(img_to_mask)
    img_to_mask = np.array(img_to_mask).reshape(1, 256, 256, 1)
    mask = mask_generation(model_seg, img_to_mask)
    mask = np.array(mask).reshape(256, 256)

    st.write("✂️ Masquage...")

    masked_img = apply_mask(
        img_original_array,
        mask,
        resize=True,
        width=img_original_array.shape[0],
        height=img_original_array.shape[1],
    )
    return mask, masked_img


def action_prediction(
    model_save_path,
    img,
    masked_value=False,
    seg_model_save_path="../../../models/cxr_reg_segmentation.best.keras",
):
    """Performs prediction on an image using a loaded model.

    This function takes an image (`img`), a path to the prediction model (`model_save_path`),
    and optionally a path to the segmentation model (`seg_model_save_path`), and a flag
    indicating whether the image is already masked (`masked_value`). It performs the
    following steps:

    1. If `masked_value` is False, it performs image pre-processing including:
        - Removing the first dimension (batch size)
        - Converting to grayscale if needed
        - Performing segmentation using the provided `seg_model_save_path`
    2. Resizes the masked image to the prediction model's expected input size.
    3. Loads the prediction model from `model_save_path`.
    4. Makes a prediction on the preprocessed image.
    5. Interprets the prediction and returns the interpreted value.

    Args:
        model_save_path: Path to the saved prediction model.
        img: The image data as a NumPy array.
        masked_value: Boolean indicating if the image is already masked (default: False).
        seg_model_save_path: Path to the segmentation model used for masking (default).

    Returns:
        The interpreted prediction value. The data type of the return value depends on
        the specific implementation of `get_predict_value`.
    """
    if not masked_value:
        st.write("👺 Masquage à faire...")
        # Remove first dimension to keep: heigth, width, channels
        img = np.array(img).reshape(img.shape[1], img.shape[2], img.shape[3])
        if img.shape[2] != 1:
            img = img[:, :, 0]
        mask, masked_img = action_masking(seg_model_save_path, img)

        # re-size masked_img to prediction model target size
        masked_img = cv2.resize(masked_img, dsize=(224, 224))
        masked_img = np.array(masked_img).reshape(1, 224, 224, 1)
        masked_img = np.repeat(masked_img[:, :, :], masked_img.shape[0], axis=3)
        img = masked_img

    st.write("⏳ Chargement du model de prediction...")
    # load model
    model = load_model(model_save_path)

    # make predict
    st.write("🤔 Prédiction...")
    pred = keras_predict_model(model, img)

    # Interpret prediction
    st.write("☝️ Interpretation...")
    pred = get_predict_value(pred)

    return pred


def unzip_images(file_uploaded, extract_path="temp"):
    pathlib.Path(extract_path).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(file_uploaded,"r") as z:
        z.extractall(extract_path)


def folder_action_prediction(
    model_save_path,
    folder_path,
    masked_value=False,
    seg_model_save_path="../../models/cxr_reg_segmentation.best.keras",
):
    
    columns = ["filename",
               "Prediction_results",
               "pred_COVID",
               "pred_Lung_Opacity",
               "pred_Normal",
               "pred_Viral_Pneumonia"]

    res_dict = {k:[] for k in columns}

    #seg_model_save_path = pathlib.Path(seg_model_save_path)
    folder_path = pathlib.Path(folder_path)

    st.write("⏳ Chargement du modèle de prédiction...")
    model = load_model(model_save_path)
    if not masked_value:
        st.write("⏳ Chargement du modèle de segmentation...")
        model_seg = load_model(seg_model_save_path)

    my_bar = st.progress(0, text="🤔 Prédiction en cours...")

    i=0
    total = len(os.listdir(folder_path))
    for radio_file in os.listdir(folder_path):
        # load image
        img = load_file(folder_path / radio_file)

        # masking if necessary
        if not masked_value:
            img_to_mask = np.array(img).reshape(img.shape[1], img.shape[2], img.shape[3])
            if img_to_mask.shape[2] != 1:
                img_to_mask = img_to_mask[:, :, 0]
            img_to_apply_mask = img_to_mask.reshape(img_to_mask.shape[0], img_to_mask.shape[1])

            img_to_mask = cv2.resize(img_to_mask, dsize=(256, 256))
            img_to_mask = scale_image(img_to_mask)
            img_to_mask = np.array(img_to_mask).reshape(1, 256, 256, 1)
            mask = mask_generation(model_seg, img_to_mask)
            mask = np.array(mask).reshape(256, 256)

            masked_img = apply_mask(
                img_to_apply_mask,
                mask,
                resize=True,
                width=img_to_apply_mask.shape[0],
                height=img_to_apply_mask.shape[1],
            )
            masked_img = cv2.resize(masked_img, dsize=(224, 224))
            masked_img = np.array(masked_img).reshape(1, 224, 224, 1)
            masked_img = np.repeat(masked_img[:, :, :], masked_img.shape[0], axis=3)
            img = masked_img

        list_pred = keras_predict_model(model, img)
        pred = get_predict_value(list_pred)
        list_pred = list_pred[0]

        res_dict["filename"].append(radio_file)
        res_dict["Prediction_results"].append(pred)
        res_dict["pred_COVID"].append(list_pred[0])
        res_dict["pred_Lung_Opacity"].append(list_pred[1])
        res_dict["pred_Normal"].append(list_pred[2])
        res_dict["pred_Viral_Pneumonia"].append(list_pred[3])

        i+=1
        # Update que toute les 5 prédictions
        if i % 5:
            percent = round((i / total) * 100)
            my_bar.progress(percent, text="🤔 Prédiction en cours...")

    my_bar.progress(100, text="👍 Prédiction fini.")

    df = pd.DataFrame(res_dict)

    return df, df.to_csv(index=False).encode('utf-8')




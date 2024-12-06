import streamlit as st
import numpy as np
import pathlib
import sys
import cv2

# To load our custom model functions
path = pathlib.Path("../../models").resolve()
sys.path.insert(0, str(path))
path = pathlib.Path("../..").resolve()
sys.path.insert(0, str(path))

## clean for reloading scriptwithout spamming sys.path insert
sys.path = list(dict.fromkeys(sys.path))

from model_functions import load_model, keras_predict_model, get_predict_value
from seg_predict_model import scale_image, mask_generation, apply_mask
from plot_functions import make_gradcam_heatmap, overlay_heatmap_on_array


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

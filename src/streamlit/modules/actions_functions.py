import streamlit as st
import io
import numpy as np
import matplotlib.cm as cm
from PIL import Image
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
from data_functions import load_resize_img_from_buffer
from plot_functions import make_gradcam_heatmap, overlay_heatmap_on_array

from modules.img_functions import convert_array_to_PIL, convert_PIL_to_io


def action_prediction(
    model_save_path,
    img,
    masked_value=False,
    seg_model_save_path="../../../models/cxr_reg_segmentation.best.keras",
):
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


def action_visualization(model_save_path, img, img_original_array, layer_name):
    # Load model
    st.write("⏳ Chargement du model...")
    model = load_model(model_save_path)
    efficientnet = model.get_layer("efficientnetb4")

    st.write("🖌️ Création de l'overlay...")
    # Make gradcam
    heatmap = make_gradcam_heatmap(img, efficientnet, layer_name)

    # Overlay the heatmap on the original image
    overlay = overlay_heatmap_on_array(heatmap, img_original_array, alpha=0.4)

    return heatmap, overlay


def action_masking(seg_model_save_path, img_original_array):
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

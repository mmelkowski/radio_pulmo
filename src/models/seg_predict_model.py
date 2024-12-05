# import system libs
import os
import pathlib

# log and command
import click

# import data handling tools
import numpy as np

# import image lib
import cv2
from PIL import Image

# import custom function
from model_functions import load_model

# Ignore Warnings
import warnings

warnings.filterwarnings("ignore")


def load_img(img_path, width=256, height=256):
    """Loads and preprocesses an image.

    This function loads an image from the specified path, resizes it to the given
    width and height, and converts it to grayscale.

    Args:
        img_path: The path to the image file.
        width: The desired width of the resized image. Defaults to 224.
        height: The desired height of the resized image. Defaults to 224.

    Returns:
    A tuple containing:
        - The original image as a NumPy array.
        - The resized and grayscale image as a NumPy array.
    """
    img_original = cv2.imread(img_path)
    # select only one channel
    img_resize = cv2.resize(img_original, (width, height))[:, :, 0]
    return img_original, img_resize


def scale_image(image):
    """Scales an image pixel values to the range of -1 to 1.

    This function takes an image as a NumPy array and scales its pixel values
    to a custom range easier for prediction.

    Args:
        image: The input image as a NumPy array.

    Returns:
        The scaled image as a NumPy array.
    """
    images_scaled = (image - 127) / 127
    return images_scaled


def mask_generation(model_seg, images_scaled):
    """Generates segmentation masks using a trained model.

    This function feeds input images through a trained segmentation model to
    obtain predicted segmentation masks. The masks are binarized based on a
    threshold of 0.5.

    Args:
        model_seg: The trained segmentation model.
        images_scaled: A NumPy array of scaled input images.

    Returns:
        A NumPy array of predicted segmentation masks.
    """
    masks = model_seg.predict(images_scaled)
    masks[masks > 0.5] = 255
    masks[masks <= 0.5] = 0
    return masks


def apply_mask(img, msk, resize=False, width=256, height=256):
    """Applies a mask to an image.

    This function applies a binary mask to an image. Optionally,
    the resulting masked image can be resized to a specified width and height.

    Args:
        img: The input image as a NumPy array.
        msk: The binary mask as a NumPy array.
        resize: Whether to resize the masked image. Defaults to False.
        width: The desired width of the resized image.
        height: The desired height of the resized image.

    Returns:
        The masked image as a NumPy array.
    """
    msk = msk.astype(img.dtype)
    if resize:
        msk = cv2.resize(msk, dsize=(width, height))
    masked_image = cv2.bitwise_and(img, msk)

    return masked_image


def images_export(save_path, masked_image, mask, img_name):
    """Exports masked images and their corresponding masks to specified paths.

    Args:
        save_path (str): Path to the directory where the images and masks will be saved.
        masked_image (np.ndarray): The masked image data as a NumPy array. The array
            should be of type `uint8`.
        mask (np.ndarray): The mask data as a NumPy array.
        img_name (str): Name of the image (without extension). This will be used
            to form the filenames for the saved images.

    Raises:
        OSError: If an error occurs while creating the directories for masked images or masks.

    Returns:
        None: This function does not explicitly return a value, but it saves a masked image
        and its mask.
    """

    save_path = pathlib.Path(save_path)
    masked_save = save_path / "masked_images"
    mask_save = save_path / "mask"

    if not os.path.exists(masked_save):
        os.makedirs(masked_save)
    if not os.path.exists(mask_save):
        os.makedirs(mask_save)

    masked_image = masked_image.astype(np.uint8)
    masked_image = Image.fromarray(masked_image)

    # Saving  masked and mask images
    masked_image.save(masked_save / f"masked_{img_name}.png")
    cv2.imwrite(mask_save / f"mask_{img_name}.png", mask)


def process_image(img_folder_path, img_filename, model, save_path, output_resize=False):
    """Processes an image by applying a segmentation mask.

    This function loads an image, resizes it, generates a segmentation mask using the provided model,
    applies the mask to the image, and saves both the masked image and the mask itself.

    Args:
        img_folder_path: Path to the folder containing the image.
        img_filename: Filename of the image to be processed.
        model: The trained segmentation model.
        save_path: Path to the directory where the processed images and masks will be saved.
        output_resize: Optional tuple specifying the desired output size of the masked image. If
                        None, the original image size is used.

    Raises:
        ValueError: If the output resize tuple is not None and has a length different from 2.

    Returns:
        None
    """
    # load image
    img_path = pathlib.Path(img_folder_path) / img_filename
    img_original, img = load_img(img_path, width=256, height=256)

    images_scaled = scale_image(img)
    images_scaled = np.array(images_scaled).reshape(1, 256, 256, 1)

    # predict mask
    mask = mask_generation(model, images_scaled)

    mask = np.array(mask).reshape(256, 256, 1)

    if output_resize:
        img_original = cv2.resize(
            img_original, dsize=(output_resize[0], output_resize[1])
        )
    mask = cv2.resize(mask, dsize=(img_original.shape[0], img_original.shape[1]))

    # apply mask
    # TODO correct to apply mask to 3channel images
    masked_image = apply_mask(img_original[:, :, 0], mask)

    # save mask and masked_img
    images_export(save_path, masked_image, mask, img_path.stem)


@click.command(context_settings={"show_default": True})
@click.option(
    "--model_name", help="Name of the segmentation model file to load.", required=True
)
@click.option(
    "--model_path",
    default="../../models",
    help="Abs or relative path to the models storage folder.",
)
@click.option(
    "--save_location",
    default="../../data/processed",
    help="Abs or relative path to the output folder save location.",
)
@click.option("--filepath", help="Path to the image to mask.")
@click.option(
    "--folderpath",
    help="Path to the folder containing images to mask.",
)
@click.option(
    "--output_height",
    help="Masked and mask output height, by default will be same height as the original.",
)
@click.option(
    "--output_width",
    help="Masked and mask output width, by default will be same width as the original.",
)
def seg_predict_model(
    model_name,
    model_path,
    save_location,
    filepath,
    folderpath,
    output_height,
    output_width,
):
    """This script predicts segmentation masks for images.

    It can handle either a single image or a folder containing multiple images.

    **usage**

    - Single image:

    python3 seg_predict_model.py --model_name <model_name>.keras --filepath <image_path>

    - Folder of images:

    python3 seg_predict_model.py --model_name <model_name>.keras --folderpath <folder_path>
    """
    if filepath and folderpath:
        raise Exception(
            "[ERROR] options --filepath and --folderpath cannot be used at the same time."
        )

    # Option to resize the outputed picture to the
    if output_height and output_width:
        output_resize = (int(output_height), int(output_width))
    else:
        output_resize = False

    # config
    save_location = pathlib.Path(save_location)
    masking_results_folder = save_location / "masking_results"
    model_save_path = pathlib.Path(model_path) / model_name

    # segmentation
    ## load model
    print("[INFO] Model Loading")
    model = load_model(model_save_path)

    if filepath:
        print("[INFO] Processing image...")
        filepath = pathlib.Path(filepath)
        img_filename = filepath.name
        process_image(
            filepath.resolve().parents[0],
            img_filename,
            model,
            masking_results_folder,
            output_resize=output_resize,
        )
        print(f"Masked Image and mask saved at: {masking_results_folder}")

    elif folderpath:
        print("[INFO] Processing folder...")
        folderpath = pathlib.Path(folderpath)
        # process images in folder
        for img_filename in os.listdir(folderpath):
            print(img_filename)
            process_image(
                folderpath,
                img_filename,
                model,
                masking_results_folder,
                output_resize=output_resize,
            )
        print(f"Masked Image and mask saved at: {masking_results_folder}")

    else:
        raise Exception("[ERROR] No options --filepath or --folderpath found.")
    
    print("[INFO] Processing Done.")


if __name__ == "__main__":
    seg_predict_model()

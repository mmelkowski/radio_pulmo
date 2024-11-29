import cv2
import os
import pathlib
import kagglehub
import shutil
import click


def download_dataset(kaggle_dataset_path, covid_dataset_name, path_to_data):
    # Download latest version
    path = kagglehub.dataset_download(kaggle_dataset_path)

    # Move to data storage location
    source = pathlib.Path(path) / covid_dataset_name
    shutil.move(source, path_to_data)


def process_data(
    folder_to_process,
    data_folder_path,
    output_path,
    target_size=(256, 256),
    smaller_set=False,
    small_size=20,
):
    """Processes images in a given folder, resizing and applying masks.

    This function iterates over images in the specified folder, resizes them to the
    target size, and applies a mask to each image. The processed images are then
    saved to the output path.

    Args:
        folder_to_process (str): The path to the folder containing the images to be processed.
        data_folder_path (str): The path to the folder containing the data for processing.
        output_path (str): The path to the folder where the processed images will be saved.
        target_size (tuple): The target size for resizing the images. Defaults to (256, 256).
        smaller_set (bool): Whether to create a smaller dataset. Defaults to False.
        small_size (int): The size of the smaller dataset. Defaults to 20 image per image type.
    """
    for img_type in folder_to_process:
        print(f"Processing folder: {img_type}")

        img_folder_path = data_folder_path / img_type / "images"
        mask_folder_path = data_folder_path / img_type / "masks"

        output_folder_path = output_path / img_type / "images"
        output_folder_path.mkdir(parents=True, exist_ok=True)

        nb_image_done = 0
        for image_name, mask_name in zip(
            os.listdir(img_folder_path), os.listdir(mask_folder_path)
        ):

            image_path = img_folder_path / image_name
            mask_path = mask_folder_path / mask_name

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # resize (skip if size already match the target)
            if target_size != (image.shape[0], image.shape[1]):
                image = cv2.resize(image, dsize=target_size)

            if target_size != (mask.shape[0], mask.shape[1]):
                mask = cv2.resize(mask, dsize=target_size)

            # masking
            res = cv2.bitwise_and(image, image, mask=mask)

            # Write masked image
            output_image_name = image_name + "_masked.png"
            cv2.imwrite(output_folder_path / output_image_name, res)

            nb_image_done += 1
            if smaller_set and nb_image_done >= small_size:
                break

        print(f"Processing folder: {img_type} done.")


@click.command(context_settings={"show_default": True})
@click.option(
    "--kaggle_dataset_path",
    default="tawsifurrahman/covid19-radiography-database",
    help="Kaggle dataset name to download.",
)
@click.option(
    "--path_to_data",
    default="../../data",
    help="Abs or relative path to the data folder where raw and process folder are expected.",
)
@click.option(
    "--covid_dataset_name",
    default="COVID-19_Radiography_Dataset",
    help="Kaggle dataset name after download.",
)
@click.option(
    "--covid_dataset_processed_name",
    default="COVID-19_masked_features",
    help="Dataset name afer processing.",
)
@click.option(
    "--folder_to_process",
    multiple=True,
    default=["Lung_Opacity", "COVID", "Normal", "Viral Pneumonia"],
    help="List of image type to process.",
)
@click.option("--target_width", default=256, help="Image width after resizing")
@click.option("--target_height", default=256, help="Image height after resizing")
@click.option(
    "--smaller_set", default="False", help="Option to produce a smaller dataset."
)
@click.option("--small_size", default=20, help="Image height after resizing")
def build_features(
    kaggle_dataset_path,
    path_to_data,
    covid_dataset_name,
    covid_dataset_processed_name,
    folder_to_process,
    target_height,
    target_width,
    smaller_set,
    small_size,
):
    """Main function to download and process the kaggle covid19-radiography-database.

    This scripts is meant to be executed in its folder with the command "python3 build_features.py".
    It will proceed to the download, extract and processing in the data folder (path_to_data argument).
    The processing consist of a re-size and masking of each image by its mask.

    The downloaded database will contain ~1go of data after extraction.
    """
    path_to_data = pathlib.Path(path_to_data)
    path_to_raw = path_to_data / "raw"
    path_to_process = path_to_data / "processed"

    if not os.path.exists(path_to_raw):
        os.makedirs(path_to_raw)
    if not os.path.exists(path_to_process):
        os.makedirs(path_to_process)

    covid_dataset_raw_path = path_to_raw / covid_dataset_name
    covid_dataset_processed_path = path_to_process / covid_dataset_processed_name

    target_size = (target_width, target_height)

    # Check if data not already present:
    if not os.path.isdir(covid_dataset_raw_path):
        print("Start kaggle dataset download...")
        download_dataset(kaggle_dataset_path, covid_dataset_name, path_to_raw)
        print(f"Download Done. Data stored at: {covid_dataset_raw_path}")
    else:
        print(
            f"Raw data already found at: {covid_dataset_raw_path}",
            "\n",
            "Download skipped.",
        )

    # Check if data not already processed:
    if not os.path.isdir(covid_dataset_processed_path):
        print("Processing raw data...")
        process_data(
            folder_to_process,
            covid_dataset_raw_path,
            covid_dataset_processed_path,
            target_size=target_size,
            smaller_set=smaller_set == "True",
            small_size=small_size,
        )
        print(f"Processing done. Data stored at: {covid_dataset_processed_path}")
    else:
        print(
            f"Processed data already found at: {covid_dataset_processed_path}",
            "\n",
            "Processing skipped.",
        )


if __name__ == "__main__":
    build_features()

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


def process_data(folder_to_process, data_folder_path, output_path, 
                 target_size=(256, 256), smaller_set=False, small_size=20):
    for img_type in  folder_to_process:
        print(f"Processing folder: {img_type}")

        img_folder_path = data_folder_path / img_type / "images"
        mask_folder_path = data_folder_path / img_type / "masks"

        output_folder_path = output_path / img_type
        output_folder_path.mkdir(parents=True, exist_ok=True)

        nb_image_done = 0
        for image_name, mask_name in zip(os.listdir(img_folder_path),
                                         os.listdir(mask_folder_path)):

            image_path = img_folder_path / image_name
            mask_path = mask_folder_path / mask_name

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # resize (skip if size already match the target)
            if target_size != (image.shape[0], image.shape[1]):
                image = cv2.resize(image, dsize = target_size)

            if target_size != (mask.shape[0], mask.shape[1]):
                mask = cv2.resize(mask, dsize = target_size)

            # masking
            res =  cv2.bitwise_and(image, image, mask=mask)

            # Write masked image
            output_image_name = image_name + '_masked.png'
            cv2.imwrite(output_folder_path / "images" / output_image_name, res)

            nb_image_done += 1
            if smaller_set and nb_image_done >= small_size:
                break

        print(f"Processing folder: {img_type} done.")


@click.command()
@click.option("--kaggle_dataset_path", default="tawsifurrahman/covid19-radiography-database", help="Kaggle dataset name to download.")
@click.option("--path_to_data", default="../../data", help="Abs or relative path to the data folder where raw and process folder are expected.")
@click.option("--covid_dataset_name", default="COVID-19_Radiography_Dataset", help="Kaggle dataset name after download.")
@click.option("--covid_dataset_processed_name", default="COVID-19_masked_features", help="Dataset name afer processing.")
@click.option("--folder_to_process", multiple=True, default=["Lung_Opacity","COVID","Normal","Viral Pneumonia"], help="List of image type to process.")
@click.option("--target_width", default=256, help="Image width after resizing")
@click.option("--target_height", default=256, help="Image height after resizing")
def build_features(kaggle_dataset_path, 
                   path_to_data,
                   covid_dataset_name,
                   covid_dataset_processed_name,
                   folder_to_process,
                   target_height,
                   target_width
                   ):
    """Function to download and process the covid19-radiography-database.
    """
    #kaggle_dataset_path = "tawsifurrahman/covid19-radiography-database"
    #folder_to_process = ["Lung_Opacity","COVID","Normal","Viral Pneumonia"]

    path_to_data = pathlib.Path(path_to_data)
    path_to_raw = path_to_data / "raw"
    path_to_process = path_to_data / "processed"

    covid_dataset_raw_path = path_to_raw / covid_dataset_name
    covid_dataset_processed_path = path_to_process / covid_dataset_processed_name

    target_size = (target_width, target_height)

    # Check if data not already present:
    if not os.path.isdir(covid_dataset_raw_path):
        print("Start kaggle dataset download...")
        download_dataset(kaggle_dataset_path, covid_dataset_name, path_to_raw)
        print(f"Download Done. Data stored at: {covid_dataset_raw_path}")
    else:
        print(f"Raw data already found at: {covid_dataset_raw_path}","\n","Download skipped.")

    # Check if data not already processed:
    if not os.path.isdir(covid_dataset_processed_path):
        print("Processing raw data...")
        process_data(folder_to_process,
                     covid_dataset_raw_path,
                     covid_dataset_processed_path,
                     target_size=target_size)
        print(f"Processing done. Data stored at: {covid_dataset_processed_path}")
    else:
        print(f"Processed data already found at: {covid_dataset_processed_path}","\n","Processing skipped.")


if __name__ == "__main__":
    build_features()
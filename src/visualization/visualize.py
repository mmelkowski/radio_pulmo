# import system libs
import os
import pathlib
import sys

# log and command
import click

# import data handling tools

# import Deep learning Libraries
from tensorflow import keras
# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")
path = pathlib.Path('../models').resolve()

sys.path.insert(0, str(path))

# custom functions
from data_functions import define_paths, define_df, split_data, create_gens, loading_dataset
from plot_functions import plot_gradcam


def load_model(model_save_path):
    #TODO should import predict_model's "load_model" function
    return keras.models.load_model(model_save_path)


@click.command()
@click.option("--model_name", help="Name of the model file to load.", required=True)
@click.option("--model_path", default="../../models",
              help="Abs or relative path to the models storage folder.")
@click.option("--path_to_data", default="../../data/processed", 
              help="Abs or relative path to the processed data folder are expected.")
@click.option("--covid_dataset_processed_name", default="COVID-19_masked_features", 
              help="Processed dataset name.")
@click.option("--save_location", default="../../models",
              help="Abs or relative path to figure save location.")
@click.option('--gradcam',
              type=click.Choice(["True", "False"]),
              default="True",
              help='Make Grad-CAM figure, only working with EfficentNetB4model.')
@click.option('--last_conv_layer_name',
              default="top_conv",
              help='Model conv layer to use for gradcam, can be "stem_conv, "block4f_expand_conv" or "top_conv", for first, middle or last layer.')
@click.option('--img_path',
              help='Image path to use for gradient.')
def visualize(
    model_name, 
    model_path,
    path_to_data,
    covid_dataset_processed_name,
    save_location,
    gradcam,
    last_conv_layer_name,
    img_path
    ):
    # config
    model_save_path = pathlib.Path(model_path) / model_name
    data_dir = pathlib.Path(path_to_data) / covid_dataset_processed_name

    # load model
    model = load_model(model_save_path)

    if gradcam == 'True':
        if not img_path:
            print("No Image path given, the gradcam will a Covid image by default.")
            img_path = data_dir / "COVID" / "images"
            img = os.listdir(img_path)[0]
            img_path = img_path / img
            print(f"Using: {img_path}.")

        efficientnet = model.get_layer("efficientnetb4")
        plot_gradcam(save_location, img_path, efficientnet, last_conv_layer_name)
        print(f"Grad-CAM saved as: {save_location}.")


if __name__ == "__main__":
    visualize()
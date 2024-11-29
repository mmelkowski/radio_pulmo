# import system libs
import pathlib

# log and command
import click

# import data handling tools
import numpy as np

# Ignore Warnings
import warnings

warnings.filterwarnings("ignore")


"""
NB import
"""
import numpy as np
import tensorflow as tf

# from tqdm import tqdm
import os
import cv2
import matplotlib.pyplot as plt
import random
import yaml

# import Deep learning Libraries
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Conv2DTranspose,
    concatenate,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    LearningRateScheduler,
    EarlyStopping,
    ReduceLROnPlateau,
)

import gc

from sklearn.model_selection import train_test_split


def check_for_GPU():
    return bool(tf.config.list_physical_devices("GPU"))


# visualise image and mask
def plotMask(X, y):
    sample = []

    for i in range(6):
        left = X[i]
        right = y[i]
        combined = np.hstack((left, right))
        sample.append(combined)

    for i in range(0, 6, 3):

        plt.figure(figsize=(25, 10))

        plt.subplot(2, 3, 1 + i)
        plt.imshow(sample[i])

        plt.subplot(2, 3, 2 + i)
        plt.imshow(sample[i + 1])

        plt.subplot(2, 3, 3 + i)
        plt.imshow(sample[i + 2])

        plt.show()


def getData(data_folder_path, X_shape=256, dim=256, N=None):
    data_folder_path = pathlib.Path(data_folder_path)
    im_array = []
    mask_array = []

    for cat in ["Normal", "Lung_Opacity", "Viral Pneumonia", "COVID"]:
        image_path = data_folder_path / cat / "images"
        mask_path = data_folder_path / cat / "masks"

        files = os.listdir(image_path)
        random.Random(1337).shuffle(files)

        if N is None:
            N = len(files)
        else:
            N = min(N, len(files))
        for i in files[:N]:
            im = cv2.resize(
                cv2.imread(os.path.join(image_path, i)), (X_shape, X_shape)
            )[:, :, 0]
            mask = cv2.resize(
                cv2.imread(os.path.join(mask_path, i)), (X_shape, X_shape)
            )[:, :, 0]
            im_array.append(im)
            mask_array.append(mask)

    images = np.array(im_array).reshape(len(im_array), dim, dim, 1)
    masks = np.array(mask_array).reshape(len(mask_array), dim, dim, 1)
    return images, masks


def to_dataset(images, masks, batch_size):
    train_vol, valid_vol, train_seg, valid_seg = train_test_split(
        (images - 127) / 127,
        (masks > 127).astype(np.float32),
        test_size=0.1,
        random_state=2018,
    )

    train_vol, test_vol, train_seg, test_seg = train_test_split(
        train_vol, train_seg, test_size=0.1, random_state=2018
    )
    return train_vol, valid_vol, test_vol, train_seg, valid_seg, test_seg


def to_datagen(
    batch_size, train_vol, valid_vol, test_vol, train_seg, valid_seg, test_seg
):
    datagen = ImageDataGenerator()
    # Augmenter respectivement les jeu de données d'entrainement
    train_dataset = datagen.flow(train_vol, train_seg, batch_size=batch_size)

    test_dataset = datagen.flow(test_vol, test_seg, batch_size=batch_size)

    val_dataset = datagen.flow(valid_vol, valid_seg, batch_size=batch_size)

    print(
        "Shape de train_scaled: ",
        train_vol.shape,
        "type de train_scaled: ",
        train_vol.dtype,
    )
    return train_dataset, test_dataset, val_dataset


# Importing data
def wrap_plotMask(size_max_cat):
    dim = 256
    images, masks = getData(dim, N=size_max_cat)
    print("visualize dataset")
    plotMask(images, masks)
    print("dataset size : images", images.shape, "masks", masks.shape)
    print(
        "Shape de image_scaled: ", images.shape, "type de image_scaled: ", images.dtype
    )

    return images, masks


def unet(input_size=(256, 256, 1)):
    inputs = Input(input_size)

    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool4)
    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv5)

    up6 = concatenate(
        [Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(conv5), conv4],
        axis=3,
    )
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(up6)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv6)

    up7 = concatenate(
        [Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(conv6), conv3],
        axis=3,
    )
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(up7)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv7)

    up8 = concatenate(
        [Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(conv7), conv2],
        axis=3,
    )
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(up8)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv8)

    up9 = concatenate(
        [Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(conv8), conv1],
        axis=3,
    )
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(up9)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv9)

    conv10 = Conv2D(1, (1, 1), activation="sigmoid")(conv9)

    return Model(inputs=[inputs], outputs=[conv10])


def get_callback_list(save_path, save_name="cxr_reg_segmentation.best.keras"):
    weight_path = pathlib.Path(save_path) / save_name
    checkpoint = ModelCheckpoint(
        weight_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min"
    )  # save_weights_only = True)
    reduceLROnPlat = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        verbose=1,
        mode="min",
        min_delta=0.001,
        cooldown=2,
        min_lr=1e-6,
    )
    early = EarlyStopping(monitor="val_loss", mode="min", min_delta=0.0001, patience=5)
    return [checkpoint, early, reduceLROnPlat]


def compile_model(display_summary=False, input_size=(256, 256, 1)):
    model = unet(input_size=input_size)
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="Dice",
        metrics=["BinaryIoU", "binary_accuracy"],
    )
    if display_summary:
        model.summary()

    return model


def clear_temp():
    gc.collect()
    tf.keras.backend.clear_session()


def training(model, train_dataset, test_dataset, batch_size, callbacks_list):
    loss_history = model.fit(
        train_dataset,
        batch_size=batch_size,
        epochs=3,
        validation_data=test_dataset,
        callbacks=callbacks_list,
    )
    return loss_history


def plot_training(loss_history, save_path, title="training_history"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(loss_history.history["loss"], "-", label="Loss")
    ax1.plot(loss_history.history["val_loss"], "-", label="Validation Loss")
    ax1.legend()

    ax2.plot(
        100 * np.array(loss_history.history["binary_accuracy"]), "-", label="Accuracy"
    )
    ax2.plot(
        100 * np.array(loss_history.history["val_binary_accuracy"]),
        "-",
        label="Validation Accuracy",
    )
    ax2.legend()

    plt.savefig(pathlib.Path(save_path) / f"{title}.png")


def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2.0 * intersection) / (np.sum(y_true) + np.sum(y_pred))


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coefficient(y_true, y_pred)
    return loss


def test_accuracy(model, X, y):
    X_test, y_test = X, y
    batch_size = 8
    predictions = model.predict(X_test, batch_size=batch_size)

    # Reshape predictions if necessary (assuming single-channel masks)
    y_pred = np.squeeze(predictions)
    y_pred = np.expand_dims(y_pred, axis=-1)

    threshold = 0.5
    y_pred_binary = (y_pred >= threshold).astype(int)

    # Calculate pixel-wise accuracy
    accuracy = np.mean(y_pred_binary == y_test)

    dice_coefficient_score = dice_coefficient(y_test, predictions)
    dice_loss_score = dice_loss(y_test, predictions)
    print(f"Dice Coefficient: {dice_coefficient_score:.4f}")
    print(f"Dice Loss: {dice_loss_score:.4f}")
    print(f"Pixel accuracy: {accuracy:.4f}")
    print("")
    return float(dice_coefficient_score), float(dice_loss_score), float(accuracy)


def print_test_accuracy(
    model_seg, train_vol, train_seg, test_vol, test_seg, valid_vol, valid_seg
):
    score_dict = {}
    print("Training dataset:")
    dice_coefficient_score, dice_loss_score, accuracy = test_accuracy(
        model_seg, train_vol[:1000], train_seg[:1000]
    )
    score_dict["train"] = {
        "dice_coefficient_score": dice_coefficient_score,
        "dice_loss_score": dice_loss_score,
        "accuracy": accuracy,
    }

    print("Test dataset:")
    dice_coefficient_score, dice_loss_score, accuracy = test_accuracy(
        model_seg, test_vol, test_seg
    )
    score_dict["test"] = {
        "dice_coefficient_score": dice_coefficient_score,
        "dice_loss_score": dice_loss_score,
        "accuracy": accuracy,
    }

    print("Validation dataset:")
    dice_coefficient_score, dice_loss_score, accuracy = test_accuracy(
        model_seg, valid_vol, valid_seg
    )
    score_dict["valid"] = {
        "dice_coefficient_score": dice_coefficient_score,
        "dice_loss_score": dice_loss_score,
        "accuracy": accuracy,
    }
    return score_dict


def save_dict_to_yaml(data, file_path):
    """Saves a Python dictionary to a YAML file.

    Args:
        data (dict): The dictionary to save.
        file_path (str): The path to the YAML file where the dictionary will be saved.
    """
    with open(file_path, "w") as myfile:
        yaml.dump(data, myfile, default_flow_style=False)


def save_model(model_name, save_path, model, acc, save_weight=False):
    """Saves a trained Keras model and optionally its weights.

    This function saves a trained Keras model to the specified location. The
    filename incorporates the model name and the achieved accuracy for easy
    identification.
    Optionally, the function can also save the model's weights to a separate file.

    Args:
        model_name: The name of the model to be saved.
        save_path: The path to the directory where the model will be saved.
        model: The Keras model instance to be saved.
        acc: The achieved accuracy of the model (float).
        save_weight: Whether to save the model's weights in a separate file.
            Defaults to False.
    """
    # Save model
    save_id = str(f'{model_name}-{"%.2f" %round(acc, 2)}.keras')
    model_save_loc = pathlib.Path(save_path) / save_id
    model.save(model_save_loc)
    print(f"model was saved as {model_save_loc}")

    # Save weights
    if save_weight:
        weight_save_id = str(f"{model_name}.weights.h5")
        weights_save_loc = pathlib.Path(save_path) / weight_save_id
        model.save_weights(weights_save_loc)
        print(f"weights were saved as {weights_save_loc}")


# @click.option("--epochs", default=2, help="Number of epoch to train on.")
# @click.option("--img_width", default=224, help="Input image width for the model")
# @click.option("--img_height", default=224, help="Input image height for the model")


@click.command(context_settings={"show_default": True})
@click.option(
    "--path_to_data",
    default="../../data/raw",
    help="Abs or relative path to the data folder where raw and process folder are expected.",
)
@click.option(
    "--covid_dataset_name",
    default="COVID-19_Radiography_Dataset",
    help="Kaggle dataset name after download.",
)
@click.option(
    "--save_path",
    default="../../models",
    help="Abs or relative path to the models storage folder.",
)
@click.option("--batch_size", default=8, help="Size of the batch.")
def train_seg_model(
    path_to_data,
    covid_dataset_name,
    save_path,
    batch_size,
):
    """Main function to train the UNet segmentation model on the kaggle covid19-radiography-database.

    This scripts is meant to be executed in its folder with the command "python3 train_seg_model.py".
    """

    # config
    path_to_data = pathlib.Path(path_to_data)
    save_path = pathlib.Path(save_path)

    print("[INFO] Loading data")
    data_folder_path = path_to_data / covid_dataset_name
    size_max_cat = 3000
    images, masks = getData(data_folder_path, N=size_max_cat)

    print("[INFO] Data generator creation")
    train_vol, valid_vol, test_vol, train_seg, valid_seg, test_seg = to_dataset(
        images, masks, batch_size
    )
    train_dataset, test_dataset, val_dataset = to_datagen(
        batch_size, train_vol, valid_vol, test_vol, train_seg, valid_seg, test_seg
    )

    print("[INFO] Compiling model")
    dim = 256
    model = compile_model()

    print("[INFO] getting callback list")
    model_save_name = "cxr_reg_segmentation.best.keras"
    callbacks_list = get_callback_list(save_path, save_name=model_save_name)

    print("[INFO] Clear temp cached data")
    clear_temp()

    print("[INFO] Training model")
    history = training(model, train_dataset, test_dataset, batch_size, callbacks_list)

    model_save_filepath = save_path / model_save_name
    print(f"[INFO] Model save as: {model_save_filepath}")

    print("[INFO] Save training history")
    # save training history
    plot_training(history, save_path, title="training_history")

    print("[INFO] Computing test accuracy")
    score_dict = print_test_accuracy(
        model, train_vol, train_seg, test_vol, test_seg, valid_vol, valid_seg
    )

    print("[INFO] Saving test accuracy")
    save_dict_to_yaml(score_dict, save_path / f"{model_save_name}_training_score.yaml")

    print("[INFO] Segmentation model training done.")


if __name__ == "__main__":
    train_seg_model()

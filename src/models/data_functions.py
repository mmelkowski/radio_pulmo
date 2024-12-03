# import system libs
import os
import pathlib

# import data handling tools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image

# import Deep learning Libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def define_paths(data_dir):
    """Generates file paths and corresponding labels from a directory structure.

    This function recursively traverses the specified directory, identifying image files and
    assigning labels based on their folder structure. It filters out unwanted files and folders,
    such as masks or hidden files.

    Args:
        data_dir: The path to the root directory containing the image data.

    Returns:
        A tuple of two lists:
            - A list of file paths to the images.
            - A list of corresponding class labels.
    """
    filepaths = []
    labels = []

    folds = os.listdir(data_dir)
    for fold in folds:
        foldpath = os.path.join(data_dir, fold)
        # check the folders from main directory. If there are another files, ignore them
        if pathlib.Path(foldpath).suffix != "":
            continue

        filelist = os.listdir(foldpath)
        for file in filelist:
            fpath = os.path.join(foldpath, file)

            # check if there are another folders
            if pathlib.Path(foldpath).suffix == "":
                # check unneeded masks
                if (
                    pathlib.Path(fpath).parts[-1] == "masks"
                    or pathlib.Path(fpath).parts[-1] == "Masks"
                    or pathlib.Path(fpath).parts[-1] == "MASKS"
                ):
                    continue

                else:
                    o_file = os.listdir(fpath)
                    for f in o_file:
                        ipath = os.path.join(fpath, f)
                        filepaths.append(ipath)
                        labels.append(fold)

            else:
                filepaths.append(fpath)
                labels.append(fold)

    return filepaths, labels


def define_df(files, classes):
    """Creates a DataFrame combining file paths and class labels.

    This function takes lists of file paths and corresponding class labels, and creates a
    Pandas DataFrame with two columns: 'filepaths' and 'labels'. This DataFrame is commonly
    used as input for image data generators.

    Args:
        files: A list of file paths.
        classes: A list of class labels corresponding to the file paths.

    Returns:
        A Pandas DataFrame with two columns: 'filepaths' and 'labels'.
    """
    Fseries = pd.Series(files, name="filepaths")
    Lseries = pd.Series(classes, name="labels")
    return pd.concat([Fseries, Lseries], axis=1)


def split_data(data_dir, seed=42):
    """Splits a dataset into training, validation, and test sets.

    This function splits a dataset into training, validation, and test sets, ensuring a stratified
    split based on class labels.

    Args:
        data_dir: The path to the directory containing the dataset.
        seed: The random seed for the splitting process.

    Returns:
        A tuple of three DataFrames:
            - The training DataFrame
            - The validation DataFrame
            - The test DataFrame
    """
    # train dataframe
    files, classes = define_paths(data_dir)
    df = define_df(files, classes)
    strat = df["labels"]
    train_df, dummy_df = train_test_split(
        df, train_size=0.8, shuffle=True, random_state=seed, stratify=strat
    )

    # valid and test dataframe
    strat = dummy_df["labels"]
    valid_df, test_df = train_test_split(
        dummy_df, train_size=0.5, shuffle=True, random_state=seed, stratify=strat
    )

    return train_df, valid_df, test_df


def create_gens(train_df, valid_df, test_df, batch_size):
    """Creates image data generators for training, validation, and testing.

    This function processes the provided DataFrames to create ImageDataGenerator instances
    for training, validation, and testing. It applies pre defined data augmentation techniques
    to the training and validation sets to improve model generalization.

    Args:
        train_df: A DataFrame containing training data, including image paths and labels.
        valid_df: A DataFrame containing validation data, including image paths and labels.
        test_df: A DataFrame containing test data, including image paths and labels.
        batch_size: The batch size for training and validation.

    Returns:
        A tuple of three ImageDataGenerator instances:
            - The training data generator
            - The validation data generator
            - The test data generator
    """
    # define model parameters
    img_size = (224, 224)
    channels = 3  # either BGR or Grayscale
    color = "rgb"
    img_shape = (img_size[0], img_size[1], channels)

    # Recommended : use custom function for test data batch size, else we can use normal batch size.
    ts_length = len(test_df)
    test_batch_size = max(
        sorted(
            [
                ts_length // n
                for n in range(1, ts_length + 1)
                if ts_length % n == 0 and ts_length / n <= 80
            ]
        )
    )
    test_steps = ts_length // test_batch_size

    # This function which will be used in image data generator for data augmentation, it just take the image and return it again.
    def scalar(img):
        return img

    tr_gen = ImageDataGenerator(
        preprocessing_function=scalar,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=5,
        fill_mode="nearest",
        brightness_range=[0.8, 1.2],
    )
    ts_gen = ImageDataGenerator(
        preprocessing_function=scalar,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=5,
        fill_mode="nearest",
        brightness_range=[0.8, 1.2],
    )

    train_gen = tr_gen.flow_from_dataframe(
        train_df,
        x_col="filepaths",
        y_col="labels",
        target_size=img_size,
        class_mode="categorical",
        color_mode=color,
        shuffle=True,
        batch_size=batch_size,
    )

    valid_gen = ts_gen.flow_from_dataframe(
        valid_df,
        x_col="filepaths",
        y_col="labels",
        target_size=img_size,
        class_mode="categorical",
        color_mode=color,
        shuffle=True,
        batch_size=batch_size,
    )

    # Note: we will use custom test_batch_size, and make shuffle= false
    test_gen = ts_gen.flow_from_dataframe(
        test_df,
        x_col="filepaths",
        y_col="labels",
        target_size=img_size,
        class_mode="categorical",
        color_mode=color,
        shuffle=False,
        batch_size=test_batch_size,
    )

    return train_gen, valid_gen, test_gen


def loading_dataset(data_dir, batch_size, seed=42):
    """Loads and preprocesses a dataset for training and validation.

    This function loads a dataset from the specified directory, splits it into training,
    validation, and test sets, and creates data generators for training and validation.

    Args:
        data_dir: The path to the directory containing the dataset.
        batch_size: The batch size for training and validation.
        seed: The random seed for data splitting and shuffling.

    Returns:
        A tuple containing:
            - The training DataFrame
            - The validation DataFrame
            - The test DataFrame
            - The training data generator
            - The validation data generator
            - The test data generator

    Raises:
        TypeError: If the input data directory is invalid.
    """
    try:
        # Get splitted data
        train_df, valid_df, test_df = split_data(data_dir)

        # Get Generators
        train_gen, valid_gen, test_gen = create_gens(
            train_df, valid_df, test_df, batch_size
        )

        return train_df, valid_df, test_df, train_gen, valid_gen, test_gen
    except TypeError:
        print("Invalid Input")


def load_file(fpath):
    """Function for loading a file as a generator for model prediciton
    """
    img = cv2.imread(fpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # re-size for model
    target_size = (224,224)
    img = cv2.resize(img, dsize=target_size)

    img = np.array(img).reshape(1, 224, 224, 3)
    return img


def load_resize_img_from_buffer(BytesIO_obj):
    """Function for loading a file as a generator for model prediciton
    """
    image = Image.open(BytesIO_obj)
    img_array = np.array(image)
    img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    # re-size for model
    target_size = (224,224)
    img = cv2.resize(img, dsize=target_size)

    img = np.array(img).reshape(1, 224, 224, 3)
    return img
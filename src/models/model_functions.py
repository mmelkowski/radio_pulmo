# import Deep learning Libraries
from tensorflow import keras
import numpy as np


def load_model(model_save_path):
    """Loads a Keras model from a specified path.

    This function loads a previously saved Keras model from the given path.

    Args:
        model_save_path: The path to the saved Keras model.

    Returns:
        The loaded Keras model.
    """
    # TODO Add support for model download from google drive
    return keras.models.load_model(model_save_path)


def keras_predict_model(model, data):
    """Use a Keras model to predict the value of data

    This function use a previously loaded Keras model to predict the value .

    Args:
        model: The Keras model.
        data: The data to predict (single image or generator).

    Returns:
        The Keras model prediction.
    """
    return model.predict(data)


def get_predict_value(prediction, labels=["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]):
    y_pred = np.argmax(prediction, axis=1)
    return labels[int(y_pred)]
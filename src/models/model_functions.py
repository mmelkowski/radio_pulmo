# import Deep learning Libraries
from tensorflow import keras


def load_model(model_save_path):
    """Loads a Keras model from a specified path.

    This function loads a previously saved Keras model from the given path.

    Args:
        model_save_path: The path to the saved Keras model.

    Returns:
        The loaded Keras model.
    """
    #TODO Add support for model download from google drive
    return keras.models.load_model(model_save_path)

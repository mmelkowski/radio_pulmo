# import system libs
import pathlib
import itertools

# import data handling tools
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet import preprocess_input


def show_images(gen):
    """
    Displays a sample of images from the provided data generator.

    Args:
        gen (data_generator): The data generator that yields batches of images
            and labels. The generator is expected to have a `class_indices`
            attribute that is a dictionary mapping class names to integer labels.

    Returns:
        None: This function does not explicitly return a value, but it
            saves a grid of sample images with their corresponding class names.
    """
    # return classes , images to be displayed
    g_dict = gen.class_indices  # defines dictionary {'class': index}
    classes = list(
        g_dict.keys()
    )  # defines list of dictionary's kays (classes), classes names : string
    images, labels = next(gen)  # get a batch size samples from the generator

    # calculate number of displayed samples
    length = len(labels)  # length of batch size
    sample = min(length, 25)  # check if sample less than 25 images

    plt.figure(figsize=(20, 20))

    for i in range(sample):
        plt.subplot(5, 5, i + 1)
        image = images[i] / 255  # scales data to range (0 - 255)
        plt.imshow(image)
        index = np.argmax(labels[i])  # get image index
        class_name = classes[index]  # get class of image
        plt.title(class_name, color="blue", fontsize=12)
        plt.axis("off")
    plt.show()


def plot_training(hist, savepath, plt_fig_size=(20, 8)):
    """
    This function take training model and plot history of accuracy and losses with the best epoch in both of them.
    """
    """Plots the training history of a model.

    Args:
        hist (dict): The training history dictionary returned by the Keras model.fit() method.
        Expected keys include "accuracy", "loss", "val_accuracy", and "val_loss".
        savepath (str): The path to save the generated plot image.
        plt_fig_size (tuple of ints, optional): The size of the figure in inches. Defaults to (20, 8).

    Returns:
        None: This function does not explicitly return a value, but it saves an image.
    """

    # Define needed variables
    tr_acc = hist.history["accuracy"]
    tr_loss = hist.history["loss"]
    val_acc = hist.history["val_accuracy"]
    val_loss = hist.history["val_loss"]
    index_loss = np.argmin(val_loss)
    val_lowest = val_loss[index_loss]
    index_acc = np.argmax(val_acc)
    acc_highest = val_acc[index_acc]
    Epochs = [i + 1 for i in range(len(tr_acc))]
    loss_label = f"best epoch= {str(index_loss + 1)}"
    acc_label = f"best epoch= {str(index_acc + 1)}"

    # Plot training history
    plt.figure(figsize=plt_fig_size)
    plt.style.use("fivethirtyeight")

    plt.subplot(1, 2, 1)
    plt.rcParams["figure.facecolor"] = "white"
    plt.plot(Epochs, tr_loss, "orange", label="Training loss")
    plt.plot(Epochs, val_loss, "deepskyblue", label="Validation loss")
    plt.scatter(index_loss + 1, val_lowest, s=150, c="blue", label=loss_label)
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(Epochs, tr_acc, "orange", label="Training Accuracy")
    plt.plot(Epochs, val_acc, "deepskyblue", label="Validation Accuracy")
    plt.scatter(index_acc + 1, acc_highest, s=150, c="blue", label=acc_label)
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout
    plt.savefig(pathlib.Path(savepath) / "training_history.png")


def plot_confusion_matrix(
    savepath,
    cm,
    classes,
    normalize=False,
    title="Confusion Matrix",
    cmap=plt.cm.Blues,
    plt_fig_size=(10, 10),
):
    """Visualizes a confusion matrix as a heatmap.

    This function plots a confusion matrix (`cm`) using matplotlib. It takes various
    arguments to customize the plot's appearance and behavior.

    Args:
        savepath: The path to save the generated confusion matrix plot. The file
                    will be named "confusion_matrix.png" within the specified path.
        cm: The confusion matrix to be visualized. This should be a 2D numpy array
                with integer values representing counts.
        classes: A list of string labels corresponding to the classes used in the
                    confusion matrix.
        normalize: Whether to normalize the confusion matrix by dividing each element
                    by the sum of its row. Defaults to False.
        title: The title to display at the top of the plot. Defaults to "Confusion Matrix".
        cmap: The colormap to use for visualizing the confusion matrix. Defaults to plt.cm.Blues.
        plt_fig_size: A tuple representing the width and height of the plot figure
                        in inches. Defaults to (10, 10).

    Prints:
        - A message indicating whether the confusion matrix is normalized or not.
        - A message indicating the path where the confusion matrix plot is saved.

    Returns:
        None: This function does not explicitly return a value, but it saves an image.
    """

    plt.figure(figsize=plt_fig_size)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.grid(visible=None)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")

    else:
        print("Confusion Matrix, Without Normalization")

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    cm_save = pathlib.Path(savepath) / "confusion_matrix.png"
    plt.savefig(cm_save)
    print(f"Confusion Matrix saved at: {cm_save}")


def process_image(img_path, target_size=(224, 224)):
    """Processes an image for input into an EfficientNet model.

    This function loads an image from the specified path, resizes it to the target size,
    converts it to a NumPy array, adds a batch dimension, and preprocesses it using
    the `preprocess_input` function, typically from the `tf.keras.applications` module.

    Args:
        img_path: The path to the image file.
        target_size: A tuple specifying the target width and height of the image.
                        Defaults to (224, 224).

    Returns:
        A NumPy array representing the preprocessed image.
    """
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess for EfficientNet
    return img_array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generates a Grad-CAM heatmap for a given image and model.

    This function creates a Grad-CAM heatmap to visualize the areas of the image that
    contribute most to a specific class prediction.

    Args:
        img_array: A NumPy array representing the input image.
        model: The Keras model to use for generating the heatmap.
        last_conv_layer_name: The name of the last convolutional layer in the model.
        pred_index: The index of the class for which to generate the heatmap. If None,
                    the class with the highest predicted probability is used.

    Returns:
        A NumPy array representing the heatmap.
    """
    # Create a model that maps the input image to the activations of the last conv layer
    # and the final predictions
    grad_model = Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Get the gradient of the predicted class wrt the output feature map of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])  # Default: highest score
        class_channel = preds[:, pred_index]

    # Gradient of the top predicted class with regard to the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Pool the gradients over all axes
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the output feature map by the pooled gradients
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def overlay_heatmap(heatmap, img_path, alpha=0.4):
    """Overlays a heatmap onto an image.

    This function loads an image, resizes a heatmap to match the image dimensions,
    and overlays the heatmap onto the image using a specified alpha value.

    Args:
        heatmap: A NumPy array representing the heatmap.
        img_path: The path to the image file.
        alpha: The transparency of the heatmap overlay. Defaults to 0.4.

    Returns:
        A NumPy array representing the image with the overlaid heatmap.
    """
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = np.array(img)

    img = img[:, :, 0]  # select only 1

    heatmap = np.uint8(255 * heatmap)  # Rescale heatmap to 0-255
    heatmap = np.expand_dims(heatmap, axis=-1)

    # Resize heatmap to match image dimensions
    heatmap = tf.image.resize(heatmap, (img.shape[0], img.shape[1])).numpy()
    heatmap = tf.keras.preprocessing.image.array_to_img(heatmap)
    heatmap = np.array(heatmap)

    # Create a heatmap overlay
    overlay = np.clip(img * (1 - alpha) + heatmap * alpha, 0, 255).astype("uint8")
    overlay = np.clip(img * (1 - alpha) + heatmap * alpha, 0, 255).astype("uint8")
    return overlay


def plot_gradcam(savepath, img_path, model, last_conv_layer_name, pred_index=None):
    """Visualizes a Grad-CAM heatmap on an image.

    This function generates a Grad-CAM heatmap for a given image and model, overlays
    the heatmap onto the original image, and displays both images side-by-side.

    Args:
        savepath: The path to save the generated plot image.
        img_path: The path to the image file.
        model: The Keras model to use for generating the heatmap.
        last_conv_layer_name: The name of the last convolutional layer in the model.
        pred_index: The index of the class for which to generate the heatmap. If None,
                    the class with the highest predicted probability is used.

    Returns:
        None: This function does not explicitly return a value, but it saves an image.
    """
    # Preprocess the image
    img_array = process_image(img_path)

    # Generate the heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index)
    # plt.savefig(pathlib.Path(savepath) / 'heatmap.png')

    # Overlay the heatmap on the original image
    overlay = overlay_heatmap(heatmap, img_path)

    # Plot original image and heatmap
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    img = tf.keras.preprocessing.image.load_img(img_path)
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM")
    plt.imshow(overlay)
    plt.axis("off")

    save_location = pathlib.Path(savepath) / "Grad-CAM.png"
    plt.savefig(save_location)
    print(f"Grad-CAM saved as: {save_location}.")

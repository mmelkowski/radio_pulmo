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
    '''
    This function take the data generator and show sample of the images
    '''

    # return classes , images to be displayed
    g_dict = gen.class_indices        # defines dictionary {'class': index}
    classes = list(g_dict.keys())     # defines list of dictionary's kays (classes), classes names : string
    images, labels = next(gen)        # get a batch size samples from the generator

    # calculate number of displayed samples
    length = len(labels)        # length of batch size
    sample = min(length, 25)    # check if sample less than 25 images

    plt.figure(figsize= (20, 20))

    for i in range(sample):
        plt.subplot(5, 5, i + 1)
        image = images[i] / 255       # scales data to range (0 - 255)
        plt.imshow(image)
        index = np.argmax(labels[i])  # get image index
        class_name = classes[index]   # get class of image
        plt.title(class_name, color= 'blue', fontsize= 12)
        plt.axis('off')
    plt.show()


def plot_training(hist, savepath, plt_fig_size=(20, 8)):
    '''
    This function take training model and plot history of accuracy and losses with the best epoch in both of them.
    '''

    # Define needed variables
    tr_acc = hist.history['accuracy']
    tr_loss = hist.history['loss']
    val_acc = hist.history['val_accuracy']
    val_loss = hist.history['val_loss']
    index_loss = np.argmin(val_loss)
    val_lowest = val_loss[index_loss]
    index_acc = np.argmax(val_acc)
    acc_highest = val_acc[index_acc]
    Epochs = [i+1 for i in range(len(tr_acc))]
    loss_label = f'best epoch= {str(index_loss + 1)}'
    acc_label = f'best epoch= {str(index_acc + 1)}'

    # Plot training history
    plt.figure(figsize= plt_fig_size)
    plt.style.use('fivethirtyeight')

    plt.subplot(1, 2, 1)
    plt.rcParams['figure.facecolor'] = 'white'
    plt.plot(Epochs, tr_loss, 'orange', label= 'Training loss')
    plt.plot(Epochs, val_loss, 'deepskyblue', label= 'Validation loss')
    plt.scatter(index_loss + 1, val_lowest, s= 150, c= 'blue', label= loss_label)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(Epochs, tr_acc, 'orange', label= 'Training Accuracy')
    plt.plot(Epochs, val_acc, 'deepskyblue', label= 'Validation Accuracy')
    plt.scatter(index_acc + 1 , acc_highest, s= 150, c= 'blue', label= acc_label)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout
    plt.savefig(pathlib.Path(savepath) / 'training_history.png')


def plot_confusion_matrix(savepath, cm, classes, normalize= False, title= 'Confusion Matrix', cmap= plt.cm.Blues, plt_fig_size=(10, 10)):
    '''
    This function plot confusion matrix method from sklearn package.
    '''

    plt.figure(figsize= plt_fig_size)
    plt.imshow(cm, interpolation= 'nearest', cmap= cmap)
    plt.title(title)
    plt.colorbar()
    plt.grid(visible=None)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation= 45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis= 1)[:, np.newaxis]
        print('Normalized Confusion Matrix')

    else:
        print('Confusion Matrix, Without Normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment= 'center', color= 'white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_save = pathlib.Path(savepath) / 'confusion_matrix.png'
    plt.savefig(cm_save)
    print(f'Confusion Matrix saved at: {cm_save}')


def process_image(img_path, target_size=(224, 224)):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess for EfficientNet
    return img_array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Helper function to generate Grad-CAM heatmap
    """
    # Create a model that maps the input image to the activations of the last conv layer
    # and the final predictions
    grad_model = Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
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
    """
    Helper function to overlay the heatmap on the image
    """
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = np.array(img)

    img = img[:, :, 0] # select only 1
    
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
    """
    Grad-CAM visualization
    """
    # Preprocess the image
    img_array = process_image(img_path)

    # Generate the heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index)
    #plt.savefig(pathlib.Path(savepath) / 'heatmap.png')

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

    plt.savefig(pathlib.Path(savepath) / 'Grad-CAM.png')
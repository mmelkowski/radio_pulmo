# import system libs
import time
import pathlib

# log and command
import click

# import data handling tools
import numpy as np

# import Deep learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers

# custom functions
from modules.data_functions import (
    define_paths,
    define_df,
    split_data,
    create_gens,
    loading_dataset,
)
from modules.plot_functions import plot_training

# Ignore Warnings
import warnings

warnings.filterwarnings("ignore")


def check_for_GPU():
    return bool(tf.config.list_physical_devices("GPU"))


class MyCallback(keras.callbacks.Callback):
    """Custom callback for monitoring training progress and early stopping.

    This callback extends the Keras `Callback` class to provide custom functionality
    for monitoring training metrics and implementing early stopping strategies.

    "ask_epoch" parameter is only used in the jupyter notebook for manual stop.

    Args:
        model: The Keras model being trained.
        patience (int): The number of epochs with no improvement in the validation loss
            before reducing the learning rate.
        stop_patience (int): The number of epochs with no improvement in the validation loss
            after reducing the learning rate before stopping training.
        threshold (float): The minimum improvement required in the validation loss to avoid
            early stopping.
        factor (float): The factor by which to reduce the learning rate.
        batches (int): The number of batches per epoch.
        epochs (int): The total number of epochs.
        ask_epoch (int): The epoch at which to ask the user if they want to continue training.
    """

    def __init__(
        self,
        model,
        patience,
        stop_patience,
        threshold,
        factor,
        batches,
        epochs,
        ask_epoch,
    ):
        super(MyCallback, self).__init__()
        # self.model = model
        self.my_model = model
        self.patience = patience  # specifies how many epochs without improvement before learning rate is adjusted
        self.stop_patience = stop_patience  # specifies how many times to adjust lr without improvement to stop training
        self.threshold = threshold  # specifies training accuracy threshold when lr will be adjusted based on validation loss
        self.factor = factor  # factor by which to reduce the learning rate
        self.batches = batches  # number of training batch to run per epoch
        self.epochs = epochs
        self.ask_epoch = ask_epoch
        self.ask_epoch_initial = (
            ask_epoch  # save this value to restore if restarting training
        )

        # callback variables
        self.count = 0  # how many times lr has been reduced without improvement
        self.stop_count = 0
        self.best_epoch = 1  # epoch with the lowest loss
        self.initial_lr = float(
            tf.keras.backend.get_value(model.optimizer.learning_rate)
        )  # get the initial learning rate and save it
        self.highest_tracc = 0.0  # set highest training accuracy to 0 initially
        self.lowest_vloss = np.inf  # set lowest validation loss to infinity initially
        self.best_weights = (
            self.my_model.get_weights()
        )  # set best weights to model's initial weights
        self.initial_weights = (
            self.my_model.get_weights()
        )  # save initial weights if they have to get restored

    # Define a function that will run when train begins
    def on_train_begin(self, logs=None):
        """# skip asking
        msg = 'Do you want model asks you to halt the training [y/n] ?'
        print(msg)
        ans = input('')
        if ans in ['Y', 'y']:
            self.ask_permission = 1
        elif ans in ['N', 'n']:
            self.ask_permission = 0
        """
        self.ask_permission = 0
        msg = "{0:^8s}{1:^10s}{2:^9s}{3:^9s}{4:^9s}{5:^9s}{6:^9s}{7:^10s}{8:10s}{9:^8s}".format(
            "Epoch",
            "Loss",
            "Accuracy",
            "V_loss",
            "V_acc",
            "LR",
            "Next LR",
            "Monitor",
            "% Improv",
            "Duration",
        )
        print(msg)
        self.start_time = time.time()

    def on_train_end(self, logs=None):
        stop_time = time.time()
        tr_duration = stop_time - self.start_time
        hours = tr_duration // 3600
        minutes = (tr_duration - (hours * 3600)) // 60
        seconds = tr_duration - ((hours * 3600) + (minutes * 60))

        msg = f"training elapsed time was {str(hours)} hours, {minutes:4.1f} minutes, {seconds:4.2f} seconds)"
        print(msg)

        # set the weights of the model to the best weights
        self.my_model.set_weights(self.best_weights)

    def on_train_batch_end(self, batch, logs=None):
        # get batch accuracy and loss
        acc = logs.get("accuracy") * 100
        loss = logs.get("loss")

        # prints over on the same line to show running batch count
        msg = "{0:20s}processing batch {1:} of {2:5s}-   accuracy=  {3:5.3f}   -   loss: {4:8.5f}".format(
            " ", str(batch), str(self.batches), acc, loss
        )
        print(msg, "\r", end="")

    def on_epoch_begin(self, epoch, logs=None):
        self.ep_start = time.time()

    # Define method runs on the end of each epoch
    def on_epoch_end(self, epoch, logs=None):
        ep_end = time.time()
        duration = ep_end - self.ep_start

        lr = float(
            tf.keras.backend.get_value(self.my_model.optimizer.learning_rate)
        )  # get the current learning rate
        current_lr = lr
        acc = logs.get("accuracy")  # get training accuracy
        v_acc = logs.get("val_accuracy")  # get validation accuracy
        loss = logs.get("loss")  # get training loss for this epoch
        v_loss = logs.get("val_loss")  # get the validation loss for this epoch

        if (
            acc < self.threshold
        ):  # if training accuracy is below threshold adjust lr based on training accuracy
            monitor = "accuracy"
            if epoch == 0:
                pimprov = 0.0
            else:
                pimprov = (
                    (acc - self.highest_tracc) * 100 / self.highest_tracc
                )  # define improvement of model progres

            if acc > self.highest_tracc:  # training accuracy improved in the epoch
                self.highest_tracc = acc  # set new highest training accuracy
                self.best_weights = (
                    self.my_model.get_weights()
                )  # training accuracy improved so save the weights
                self.count = 0  # set count to 0 since training accuracy improved
                self.stop_count = 0  # set stop counter to 0
                if v_loss < self.lowest_vloss:
                    self.lowest_vloss = v_loss
                self.best_epoch = (
                    epoch + 1
                )  # set the value of best epoch for this epoch

            else:
                # training accuracy did not improve check if this has happened for patience number of epochs
                # if so adjust learning rate
                if self.count >= self.patience - 1:  # lr should be adjusted
                    lr = lr * self.factor  # adjust the learning by factor
                    # tf.keras.backend.set_value(self.my_model.optimizer.learning_rate, lr) # set the learning rate in the optimizer
                    self.my_model.optimizer.learning_rate = (
                        lr  # set the learning rate in the optimizer
                    )
                    self.count = 0  # reset the count to 0
                    self.stop_count = (
                        self.stop_count + 1
                    )  # count the number of consecutive lr adjustments
                    self.count = 0  # reset counter
                    if v_loss < self.lowest_vloss:
                        self.lowest_vloss = v_loss
                else:
                    self.count = self.count + 1  # increment patience counter

        else:  # training accuracy is above threshold so adjust learning rate based on validation loss
            monitor = "val_loss"
            if epoch == 0:
                pimprov = 0.0

            else:
                pimprov = (self.lowest_vloss - v_loss) * 100 / self.lowest_vloss

            if v_loss < self.lowest_vloss:  # check if the validation loss improved
                self.lowest_vloss = (
                    v_loss  # replace lowest validation loss with new validation loss
                )
                self.best_weights = (
                    self.my_model.get_weights()
                )  # validation loss improved so save the weights
                self.count = 0  # reset count since validation loss improved
                self.stop_count = 0
                self.best_epoch = (
                    epoch + 1
                )  # set the value of the best epoch to this epoch

            else:  # validation loss did not improve
                if self.count >= self.patience - 1:  # need to adjust lr
                    lr = lr * self.factor  # adjust the learning rate
                    self.stop_count = (
                        self.stop_count + 1
                    )  # increment stop counter because lr was adjusted
                    self.count = 0  # reset counter
                    # tf.keras.backend.set_value(self.my_model.optimizer.learning_rate, lr) # set the learning rate in the optimizer
                    self.my_model.optimizer.learning_rate = (
                        lr  # set the learning rate in the optimizer
                    )

                else:
                    self.count = self.count + 1  # increment the patience counter

                if acc > self.highest_tracc:
                    self.highest_tracc = acc

        msg = f"{str(epoch + 1):^3s}/{str(self.epochs):4s} {loss:^9.3f}{acc * 100:^9.3f}{v_loss:^9.5f}{v_acc * 100:^9.3f}{current_lr:^9.5f}{lr:^9.5f}{monitor:^11s}{pimprov:^10.2f}{duration:^8.2f}"
        print(msg)

        if (
            self.stop_count > self.stop_patience - 1
        ):  # check if learning rate has been adjusted stop_count times with no improvement
            msg = f" training has been halted at epoch {epoch + 1} after {self.stop_patience} adjustments of learning rate with no improvement"
            print(msg)
            self.my_model.stop_training = True  # stop training

        else:
            if self.ask_epoch != None and self.ask_permission != 0:
                if epoch + 1 >= self.ask_epoch:
                    msg = "enter H to halt training or an integer for number of epochs to run then ask again"
                    print(msg)

                    ans = input("")
                    if ans == "H" or ans == "h":
                        msg = f"training has been halted at epoch {epoch + 1} due to user input"
                        print(msg)
                        self.my_model.stop_training = True  # stop training

                    else:
                        try:
                            ans = int(ans)
                            self.ask_epoch += ans
                            msg = f" training will continue until epoch {str(self.ask_epoch)}"
                            print(msg)
                            msg = "{0:^8s}{1:^10s}{2:^9s}{3:^9s}{4:^9s}{5:^9s}{6:^9s}{7:^10s}{8:10s}{9:^8s}".format(
                                "Epoch",
                                "Loss",
                                "Accuracy",
                                "V_loss",
                                "V_acc",
                                "LR",
                                "Next LR",
                                "Monitor",
                                "% Improv",
                                "Duration",
                            )
                            print(msg)

                        except Exception:
                            print("Invalid")


def model_structure(img_size, channels, train_gen):
    # Create Model Structure
    # img_size = (224, 224)
    # channels = 3
    img_shape = (img_size[0], img_size[1], channels)
    class_count = len(
        list(train_gen.class_indices.keys())
    )  # to define number of classes in dense layer
    return img_shape, class_count


def model_choice(model_name, class_count, img_shape, display_summary=False):
    """Loads a specified model architecture for image classification.

    This function creates a Keras model based on the provided `model_name`.
    It configures the model with the given `class_count` and `img_shape`.

    Args:
        model_name (str): The name of the model architecture to load.
        class_count (int): The number of classes for the classification task.
        img_shape (tuple): The shape of the input images.
        display_summary (bool): Whether to print a summary of the model architecture. Defaults to False.

    Returns:
        A Keras model instance.
    """
    match model_name:
        case "VGG19":
            base_model = tf.keras.applications.vgg19.VGG19(
                include_top=False,
                weights="imagenet",
                input_shape=img_shape,
                pooling="max",
            )
        case "EfficientNetB0":
            base_model = tf.keras.applications.efficientnet.EfficientNetB0(
                include_top=False,
                weights="imagenet",
                input_shape=img_shape,
                pooling="max",
            )
        case "EfficientNetB4":
            base_model = tf.keras.applications.efficientnet.EfficientNetB4(
                include_top=False,
                weights="imagenet",
                input_shape=img_shape,
                pooling="max",
            )
        case "DenseNet169":
            base_model = tf.keras.applications.densenet.DenseNet169(
                include_top=False,
                weights="imagenet",
                input_shape=img_shape,
                pooling="max",
            )
        case "ResNet101":
            base_model = tf.keras.applications.resnet.ResNet101(
                include_top=False,
                weights="imagenet",
                input_shape=img_shape,
                pooling="max",
            )

    if model_name in ["EfficientNetB0", "EfficientNetB4"]:
        model = Sequential(
            [
                base_model,
                BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
                Dense(
                    256,
                    kernel_regularizer=regularizers.l2(l2=0.016),
                    activity_regularizer=regularizers.l1(0.006),
                    bias_regularizer=regularizers.l1(0.006),
                    activation="relu",
                ),
                Dropout(rate=0.45, seed=123),
                Dense(class_count, activation="softmax"),
            ]
        )
    elif model_name in ["VGG19", "DenseNet169", "ResNet101"]:
        model = Sequential([base_model, Dense(class_count, activation="softmax")])
    else:
        print("[ERROR]: Incorrect model name")

    model.compile(
        Adamax(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    if display_summary:
        print(model.summary())

    return model


def set_callback_parameters(train_gen, model, batch_size, epochs):
    """Sets parameters for the custom callback and creates a callback instance.

    This function configures parameters for the `MyCallback` class, which is used to
    monitor training progress and implement early stopping strategies.
    It then creates an instance of the `MyCallback` class with the specified parameters.

    Args:
        train_gen: The training data generator.
        model: The Keras model being trained.
        batch_size: The batch size for training.
        epochs: The total number of epochs.

    Returns:
        A list containing the `MyCallback` instance.
    """
    # batch_size = 16   # set batch size for training
    # epochs = 100   # number of all epochs in training
    patience = (
        3  # number of epochs to wait to adjust lr if monitored value does not improve
    )
    stop_patience = 10  # number of epochs to wait before stopping training if monitored value does not improve
    threshold = 0.9  # if train accuracy is < threshold adjust monitor accuracy, else monitor validation loss
    factor = 0.5  # factor to reduce lr by
    ask_epoch = 5  # number of epochs to run before asking if you want to halt training
    batches = int(
        np.ceil(len(train_gen.labels) / batch_size)
    )  # number of training batch to run per epoch

    callbacks = [
        MyCallback(
            model=model,
            patience=patience,
            stop_patience=stop_patience,
            threshold=threshold,
            factor=factor,
            batches=batches,
            epochs=epochs,
            ask_epoch=ask_epoch,
        )
    ]

    return callbacks


def launch_training(model, train_gen, epochs, callbacks, valid_gen):
    """Launches the training process for a given model using a training generator,
    epochs, callbacks, and an optional validation generator.

    Args:
        model (keras.Model): The Keras model to be trained.
        train_gen (Sequence): A data generator object that yields training data in batches.
        epochs (int): The number of epochs to train the model for.
        callbacks (List[keras.callbacks.Callback]): A list of Keras callbacks to be applied during training.
        valid_gen (Sequence): A data generator object that yields validation data in batches.

    Returns:
        history (History): A Keras History object containing training and validation logs.
    """
    # TODO add print of training argument
    history = model.fit(
        x=train_gen,
        epochs=epochs,
        verbose=0,
        callbacks=callbacks,
        validation_data=valid_gen,
        validation_steps=None,
        shuffle=False,
    )
    return history


def get_acc(model, test_df, test_gen):
    ts_length = len(test_df)
    test_batch_size = test_batch_size = max(
        sorted(
            [
                ts_length // n
                for n in range(1, ts_length + 1)
                if ts_length % n == 0 and ts_length / n <= 80
            ]
        )
    )
    test_steps = ts_length // test_batch_size
    test_score = model.evaluate(test_gen, steps=test_steps, verbose=1)
    acc = test_score[1] * 100
    return acc


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


@click.command(context_settings={"show_default": True})
@click.option(
    "--path_to_data",
    default="../../data/processed",
    help="Abs or relative path to the data folder where raw and process folder are expected.",
)
@click.option(
    "--covid_dataset_processed_name",
    default="COVID-19_masked_features",
    help="Dataset name afer processing.",
)
@click.option(
    "--model_name",
    type=click.Choice(
        ["VGG19", "EfficientNetB0", "EfficientNetB4", "DenseNet169", "ResNet101"]
    ),
    default="EfficientNetB4",
    help="Choose the model architecture.",
)
@click.option("--img_width", default=224, help="Input image width for the model")
@click.option("--img_height", default=224, help="Input image height for the model")
@click.option(
    "--save_path",
    default="../../models",
    help="Abs or relative path to the models storage folder.",
)
@click.option("--batch_size", default=16, help="Size of the batch.")
@click.option("--epochs", default=2, help="Number of epoch to train on.")
def train_model(
    path_to_data,
    covid_dataset_processed_name,
    model_name,
    img_width,
    img_height,
    save_path,
    batch_size,
    epochs,
):
    """Main function to train the model on the kaggle covid19-radiography-database.

    This scripts is meant to be executed in its folder with the command "python3 train_model.py".

    The model chose in this project is EfficientNetB4, but other model are availlable and compatible with this code:
    ['VGG19', 'EfficientNetB0', 'EfficientNetB4', 'DenseNet169', 'ResNet101']

    The model after training will be saved with its metrics in the "data/model" folder (save_path argument).

    The training of EfficientNetB4 on the dataset can take up to 45 minutes using a "GeForce RTX4070 SUPER" GPU.

    The training code was taken and adapted from Ahmed Hafez's work
    "https://www.kaggle.com/code/ahmedtronic/covid-19-radiology-vgg19-f1-score-95".
    """
    # config
    data_dir = pathlib.Path(path_to_data) / covid_dataset_processed_name

    # data loading
    print("[INFO] Data loading")
    train_df, valid_df, test_df, train_gen, valid_gen, test_gen = loading_dataset(
        data_dir, batch_size
    )

    # training
    print("[INFO] Model Definition")
    img_size = (img_width, img_height)
    channels = 3
    img_shape, class_count = model_structure(img_size, channels, train_gen)

    model = model_choice(model_name, class_count, img_shape)

    callbacks = set_callback_parameters(train_gen, model, batch_size, epochs)

    print("[INFO] Model training")
    history = launch_training(model, train_gen, epochs, callbacks, valid_gen)
    print("[INFO] Model training done")

    # saving
    print("[INFO] Model Saving.")
    acc = get_acc(model, test_df, test_gen)
    save_model(model_name, save_path, model, acc, save_weight=False)
    plot_training(history, save_path, plt_fig_size=(20, 8))


if __name__ == "__main__":
    train_model()

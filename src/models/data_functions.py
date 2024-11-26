# import system libs
import os
import pathlib

# import data handling tools
import pandas as pd
from sklearn.model_selection import train_test_split

# import Deep learning Libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def define_paths(data_dir):
    """
    Generate data paths with labels
    """
    filepaths = []
    labels = []

    folds = os.listdir(data_dir)
    for fold in folds:
        foldpath = os.path.join(data_dir, fold)
        # check the folders from main directory. If there are another files, ignore them
        if pathlib.Path(foldpath).suffix != '':
            continue

        filelist = os.listdir(foldpath)
        for file in filelist:
            fpath = os.path.join(foldpath, file)

            # check if there are another folders
            if pathlib.Path(foldpath).suffix == '':
                # check unneeded masks
                if pathlib.Path(fpath).parts[-1] == 'masks' or pathlib.Path(fpath).parts[-1] == 'Masks' or pathlib.Path(fpath).parts[-1] == 'MASKS':
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
    """
    Concatenate data paths with labels into one dataframe (to later be fitted into the model)
    """
    Fseries = pd.Series(files, name= 'filepaths')
    Lseries = pd.Series(classes, name='labels')
    return pd.concat([Fseries, Lseries], axis= 1)


def split_data(data_dir, seed=42):
    """
    Split dataframe to train, valid, and test
    """
    # train dataframe
    files, classes = define_paths(data_dir)
    df = define_df(files, classes)
    strat = df['labels']
    train_df, dummy_df = train_test_split(df, train_size= 0.8, shuffle= True, random_state= seed, stratify= strat)

    # valid and test dataframe
    strat = dummy_df['labels']
    valid_df, test_df = train_test_split(dummy_df, train_size= 0.5, shuffle= True, random_state= seed, stratify= strat)

    return train_df, valid_df, test_df


def create_gens (train_df, valid_df, test_df, batch_size):
    """
    This function takes train, validation, and test dataframe and fit them into image data generator, because model takes data from image data generator.
    Image data generator converts images into tensors. 
    """


    # define model parameters
    img_size = (224, 224)
    channels = 3 # either BGR or Grayscale
    color = 'rgb'
    img_shape = (img_size[0], img_size[1], channels)

    # Recommended : use custom function for test data batch size, else we can use normal batch size.
    ts_length = len(test_df)
    test_batch_size = max(sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length%n == 0 and ts_length/n <= 80]))
    test_steps = ts_length // test_batch_size

    # This function which will be used in image data generator for data augmentation, it just take the image and return it again.
    def scalar(img):
        return img

    tr_gen = ImageDataGenerator(preprocessing_function= scalar, horizontal_flip= True,
                                width_shift_range=0.1, 
                                height_shift_range=0.1,
                                rotation_range=5, fill_mode='nearest',
                                brightness_range=[0.8,1.2],
                                )
    ts_gen = ImageDataGenerator(preprocessing_function= scalar,
                                width_shift_range=0.1, 
                                height_shift_range=0.1,
                                rotation_range=5, fill_mode='nearest',
                                brightness_range=[0.8,1.2],
                                )

    train_gen = tr_gen.flow_from_dataframe(train_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                        color_mode= color, shuffle= True, batch_size= batch_size)

    valid_gen = ts_gen.flow_from_dataframe(valid_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                        color_mode= color, shuffle= True, batch_size= batch_size)

    # Note: we will use custom test_batch_size, and make shuffle= false
    test_gen = ts_gen.flow_from_dataframe(test_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                        color_mode= color, shuffle= False, batch_size= test_batch_size)

    return train_gen, valid_gen, test_gen


def loading_dataset(data_dir, batch_size, seed=42):
    #data_dir = '/home/tylio/code/Project_radio_pulmo/code/radio_pulmo/data/processed/covid_19_masked_copy'
    try:

        # Get splitted data
        train_df, valid_df, test_df = split_data(data_dir)

        # Get Generators
        train_gen, valid_gen, test_gen = create_gens(train_df, valid_df, test_df, batch_size)

        return train_df, valid_df, test_df, train_gen, valid_gen, test_gen
    except TypeError:
        print('Invalid Input')
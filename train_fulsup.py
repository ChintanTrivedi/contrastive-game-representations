import argparse
import json
import os

import pandas as pd
import tensorflow as tf


# Reference: https://github.com/sayakpaul/Supervised-Contrastive-Learning-in-TensorFlow-2

def parse_args():
    parser = argparse.ArgumentParser()

    # train
    parser.add_argument('--img_shape', default=224, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--model_save_directory', default='./models/fulsup', type=str)

    # dataset
    parser.add_argument('--dataset_directory', default='./datasets/Sports10', type=str)
    parser.add_argument('--train_split_file', default='./datasets/Sports10/train_split.csv', type=str)
    parser.add_argument('--val_split_file', default='./datasets/Sports10/val_split.csv', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # hyperparams for training
    batch_size = args.batch_size
    img_shape = args.img_shape

    model_save_directory = args.model_save_directory
    if not os.path.exists(model_save_directory):
        os.makedirs(model_save_directory)

    # get train and val splits
    train_df = pd.read_csv(args.train_split_file)
    val_df = pd.read_csv(args.val_split_file)

    # get training dataset from directory with image augmentations
    train_image_aug = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, zoom_range=0.05,
                                                                      horizontal_flip=True,
                                                                      width_shift_range=0.1, height_shift_range=0.1,
                                                                      rotation_range=20, brightness_range=[0.8, 1.1])
    train_generator = train_image_aug.flow_from_dataframe(train_df, directory=args.dataset_directory,
                                                          target_size=(img_shape, img_shape),
                                                          batch_size=batch_size, class_mode='sparse', shuffle=True)
    train_ds = tf.data.Dataset.from_generator(lambda: train_generator, output_types=(tf.float32, tf.float32))
    train_ds = train_ds.prefetch(buffer_size=128)
    steps_per_train_epoch = int(train_generator.n / batch_size)
    num_class_train = len(train_generator.class_indices)

    # get validation dataset from directory without image augmentations
    val_image_aug = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    val_generator = val_image_aug.flow_from_dataframe(val_df, directory=args.dataset_directory,
                                                      target_size=(img_shape, img_shape), batch_size=batch_size,
                                                      class_mode='sparse', shuffle=True)
    val_ds = tf.data.Dataset.from_generator(lambda: val_generator, output_types=(tf.float32, tf.float32))
    val_ds = val_ds.prefetch(buffer_size=128)
    steps_per_val_epoch = int(val_generator.n / batch_size)

    # Encoder Network
    def create_encoder():
        inputs = tf.keras.layers.Input((img_shape, img_shape, 3))
        encoder = tf.keras.applications.ResNet50V2(weights='imagenet', include_top=False, pooling='avg',
                                                   input_shape=(img_shape, img_shape, 3))
        encoder.trainable = True
        embeddings = encoder(inputs, training=True)
        encoder_network = tf.keras.models.Model(inputs, embeddings)
        return encoder_network

    # Genre Classifier Model
    def create_classifier(encoder, num_class=10, trainable=True):
        for layer in encoder.layers:
            layer.trainable = trainable

        inputs = tf.keras.layers.Input(shape=(img_shape, img_shape, 3))
        features = encoder(inputs)
        features = tf.keras.layers.Dropout(0.5)(features)
        features = tf.keras.layers.Dense(512, activation="relu")(features)
        features = tf.keras.layers.Dropout(0.5)(features)
        outputs = tf.keras.layers.Dense(num_class, activation="softmax")(features)

        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(tf.keras.experimental.CosineDecay(
                initial_learning_rate=args.learning_rate, decay_steps=1000)),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
        return model

    resnet_encoder = create_encoder()
    genre_classifier = create_classifier(resnet_encoder, num_class=num_class_train, trainable=True)
    genre_classifier.summary()

    training_history = genre_classifier.fit(train_ds, batch_size=batch_size, epochs=args.epochs,
                                            validation_data=val_ds,
                                            steps_per_epoch=steps_per_train_epoch,
                                            validation_steps=steps_per_val_epoch,
                                            callbacks=[tf.keras.callbacks.EarlyStopping()])

    # save training statistics and model to save directory
    json.dump(training_history.history, open(model_save_directory + '/fulsup_training_history.json', 'w'))
    genre_classifier.save(model_save_directory + '/genre_classifier.h5', overwrite=True)


if __name__ == '__main__':
    # change keras gpu options to allow memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    main()

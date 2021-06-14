import argparse
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

import losses


def parse_args():
    parser = argparse.ArgumentParser()

    # train
    parser.add_argument('--img_shape', default=224, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--model_save_directory', default='./models/supcon', type=str)

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
    train_iter = iter(train_ds)
    steps_per_train_epoch = int(train_generator.n / batch_size)
    num_class_train = len(train_generator.class_indices)

    # get validation dataset from directory without image augmentations
    val_image_aug = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    val_generator = val_image_aug.flow_from_dataframe(val_df, directory=args.dataset_directory,
                                                      target_size=(img_shape, img_shape), batch_size=batch_size,
                                                      class_mode='sparse', shuffle=True)
    val_ds = tf.data.Dataset.from_generator(lambda: val_generator, output_types=(tf.float32, tf.float32))
    val_ds = val_ds.prefetch(buffer_size=128)
    val_iter = iter(val_ds)
    steps_per_val_epoch = int(val_generator.n / batch_size)

    # Reference: https://github.com/wangz10/contrastive_loss/blob/master/model.py
    class UnitNormLayer(tf.keras.layers.Layer):
        def __init__(self):
            super(UnitNormLayer, self).__init__()

        def call(self, input_tensor):
            norm = tf.norm(input_tensor, axis=1)
            return input_tensor / (tf.reshape(norm, [-1, 1]) + 1e-19)

    # Encoder Network
    def create_encoder():
        inputs = tf.keras.layers.Input((img_shape, img_shape, 3))
        normalization_layer = UnitNormLayer()
        encoder = tf.keras.applications.ResNet50V2(weights='imagenet', include_top=False, pooling='avg',
                                                   input_shape=(img_shape, img_shape, 3))
        encoder.trainable = True
        embeddings = encoder(inputs, training=True)
        norm_embeddings = normalization_layer(embeddings)
        encoder_network = tf.keras.models.Model(inputs, norm_embeddings)
        return encoder_network

    # Projector Network
    def create_projector():
        projector = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            UnitNormLayer()
        ])
        return projector

    # Genre Classifier Model
    def create_classifier(encoder, num_class=10, trainable=True):
        for layer in encoder.layers:
            layer.trainable = trainable

        inputs = tf.keras.Input(shape=(img_shape, img_shape, 3))
        features = encoder(inputs)
        features = tf.keras.layers.Dropout(0.5)(features)
        features = tf.keras.layers.Dense(512, activation="relu")(features)
        features = tf.keras.layers.Dropout(0.5)(features)
        outputs = tf.keras.layers.Dense(num_class, activation="softmax")(features)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(tf.keras.experimental.CosineDecay(
                initial_learning_rate=args.learning_rate, decay_steps=1000)),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
        return model

    resnet_encoder = create_encoder()
    linear_projector = create_projector()
    optimizer = tf.keras.optimizers.Adam(tf.keras.experimental.CosineDecay(
        initial_learning_rate=args.learning_rate, decay_steps=1000))

    # see losses.py for other contrastive loss functions
    def compute_contrastive_loss(images, labels, train_mode=False):
        r = resnet_encoder(images, training=train_mode)
        z = linear_projector(r, training=train_mode)
        return losses.max_margin_contrastive_loss(z, labels)

    # update weights of the encoder in this step
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            loss = compute_contrastive_loss(images, labels, train_mode=True)

        gradients = tape.gradient(loss,
                                  resnet_encoder.trainable_variables + linear_projector.trainable_variables)
        optimizer.apply_gradients(zip(gradients,
                                      resnet_encoder.trainable_variables + linear_projector.trainable_variables))
        return loss

    # create empty vars to store training statistics
    con_train_loss_history = []
    con_val_loss_history = []
    sup_training_history = {'val_sparse_categorical_accuracy': [], 'val_loss': [],
                            'sparse_categorical_accuracy': [], 'loss': []}

    # start training - each epoch is 2 steps - contrastive pre-training followed by
    # supervised training with frozen encoder
    for epoch in range(args.epochs):
        # In step 1 (pre-training), train the encoder using contrastive loss
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_val_loss_avg = tf.keras.metrics.Mean()

        # Train Over MiniBatch
        for _ in tqdm(range(steps_per_train_epoch)):
            (images, labels) = next(train_iter)
            # since contrastive loss depends on batch size, only train on full batches
            if images.shape[0] == batch_size:
                loss = train_step(images, labels)
                epoch_loss_avg.update_state(loss)
        con_train_loss_history.append(epoch_loss_avg.result())

        # Get Validation Loss
        for _ in tqdm(range(steps_per_val_epoch)):
            (images, labels) = next(val_iter)
            # since contrastive loss depends on batch size, only calculate loss for full batches
            if images.shape[0] == batch_size:
                epoch_val_loss = compute_contrastive_loss(images, labels)
                epoch_val_loss_avg.update_state(epoch_val_loss)
        con_val_loss_history.append(epoch_val_loss_avg.result())

        # In step 2 (supervised), freeze weights of the encoder and train the representation to predict classes
        genre_classifier = create_classifier(resnet_encoder, num_class=num_class_train, trainable=False)
        supcon_history = genre_classifier.fit(train_ds, batch_size=batch_size, epochs=1, validation_data=val_ds,
                                              steps_per_epoch=steps_per_train_epoch,
                                              validation_steps=steps_per_val_epoch)
        sup_training_history['val_sparse_categorical_accuracy'].append(
            supcon_history.history['val_sparse_categorical_accuracy'][0])
        sup_training_history['val_loss'].append(supcon_history.history['val_loss'][0])
        sup_training_history['sparse_categorical_accuracy'].append(
            supcon_history.history['sparse_categorical_accuracy'][0])
        sup_training_history['loss'].append(supcon_history.history['loss'][0])

        print("Epoch: {}  |  Loss: {:.5f}  |  Val Loss: {:.5f}"
              .format(epoch, epoch_loss_avg.result(), epoch_val_loss_avg.result()))

    # save training metrics and trained model for contrastive pre-training and supervised training
    np.savetxt(model_save_directory + '/con_pretraining_history.txt',
               np.stack([con_train_loss_history, con_val_loss_history]), fmt='%.5f')
    json.dump(sup_training_history, open(model_save_directory + '/sup_training_history.json', 'w'))
    genre_classifier.save(model_save_directory + '/genre_classifier.h5', overwrite=True)


if __name__ == '__main__':
    # change keras gpu options to allow memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    main()

from tensorflow import keras
from tensorflow.keras import layers

LEARNING_RATE = 0.001
SEED = 42


def add_classifier(base, input_shape, name, preprocessing):
    """Shared small classifier; every experiment trains the same head design."""
    base.trainable = False
    inputs = keras.Input(shape=input_shape)
    x = layers.RandomFlip("horizontal", seed=SEED)(inputs)
    x = layers.RandomRotation(0.05, seed=SEED)(x)
    if preprocessing == "caffe":
        # VGG16 / ResNet101: RGB -> BGR and subtract ImageNet channel means.
        x = x[..., ::-1]
        x = layers.Normalization(axis=-1, mean=[103.939, 116.779, 123.68],
                                 variance=[1., 1., 1.])(x)
    elif preprocessing == "mobilenet":
        x = layers.Rescaling(1. / 127.5, offset=-1)(x)
    # EfficientNet already includes its own normalization; keep 0-255 input.
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3, seed=SEED)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name=name)
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


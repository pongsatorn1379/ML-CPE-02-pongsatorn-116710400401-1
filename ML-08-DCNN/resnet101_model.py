from tensorflow import keras
from model_builder import add_classifier


def build_model(input_shape=(224, 224, 3), weights="imagenet"):
    base = keras.applications.ResNet101(include_top=False, weights=weights, input_shape=input_shape)
    return add_classifier(base, input_shape, "ResNet101", "caffe")


if __name__ == "__main__":
    from main import main
    main(default_model="ResNet101")


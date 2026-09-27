"""สร้างและฝึก CNN แบบพื้นฐานด้วย Keras."""
from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape, num_classes, conv_layers=1, neurons=32):
    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))
    for i in range(conv_layers):
        filters = 16 if i == 0 else 32
        model.add(layers.Conv2D(filters, 3, padding="same", activation="relu"))
        model.add(layers.MaxPooling2D())

    model.add(layers.Flatten())
    model.add(layers.Dense(neurons, activation="relu"))
    model.add(layers.Dense(num_classes, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs):
    # ไม่มี callbacks เพื่อให้ฝึกครบ epochs ที่ต้องการเปรียบเทียบ
    return model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=8, verbose=1,
    )


def predict_model(model, X_test):
    probabilities = model.predict(X_test, verbose=0)
    return probabilities.argmax(axis=1), probabilities.max(axis=1)

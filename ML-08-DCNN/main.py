import argparse
import importlib
import os
import time
from pathlib import Path

# Settings are here; config.py is no longer needed.
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "outputs"
IMG_SIZE = 224
BATCH_SIZE = 8
SEED = 42
MODELS = ["VGG16", "ResNet101", "MobileNetV2", "EfficientNetB0"]
EPOCHS = [5, 10, 20]
MODEL_MODULES = {"VGG16": "vgg16_model", "ResNet101": "resnet101_model",
                 "MobileNetV2": "mobilenetv2_model", "EfficientNetB0": "efficientnetb0_model"}


def run_experiments(model_names, epochs_list):
    import tensorflow as tf
    from tensorflow import keras
    from split_data import prepare_data
    from evaluate import evaluate_model, plot_history
    from compare_results import save_comparison, save_best_model

    gpus = tf.config.list_physical_devices("GPU")
    print("TensorFlow:", tf.__version__)
    print("GPU devices:", gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if not gpus:
        print("No TensorFlow GPU detected. Training will use CPU.")

    # Split once, then reuse these same arrays for every model and epoch count.
    X_train, X_val, X_test, y_train, y_val, y_test, classes = prepare_data(DATA_PATH, IMG_SIZE, SEED)
    if len(classes) != 2:
        raise ValueError("This lab code expects two classes: BMW and Honda.")
    results = []
    best_dir = OUTPUT_DIR / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    for name in model_names:
        for epochs in epochs_list:
            output = OUTPUT_DIR / name / f"epochs_{epochs}"
            output.mkdir(parents=True, exist_ok=True)
            keras.backend.clear_session()
            keras.utils.set_random_seed(SEED)
            module = importlib.import_module(MODEL_MODULES[name])
            print(f"\n=== {name}: {epochs} epochs ===")
            model = module.build_model((IMG_SIZE, IMG_SIZE, 3))
            model.summary()
            start = time.perf_counter()
            # Train exactly the requested epochs; start a fresh model each time.
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                epochs=epochs, batch_size=BATCH_SIZE, verbose=2)
            seconds = time.perf_counter() - start
            val_loss, val_accuracy = model.evaluate(X_val, y_val, batch_size=BATCH_SIZE, verbose=0)
            test_loss, _ = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=0)
            probabilities = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).ravel()
            accuracy = evaluate_model(y_test, probabilities, classes, X_test, output)
            plot_history(history, output)
            results.append((name, epochs, val_accuracy, val_loss, accuracy, test_loss, seconds))
            # Keep comparison results from completed experiments if a later one stops.
            save_comparison(results, OUTPUT_DIR / "comparison")
            save_best_model(results, best_dir)
            print(f"Saved the three PNG images to: {output}")
            del model
            keras.backend.clear_session()

    print("\n=== Comparison ===")
    print(f"{'Model':<17} {'Epochs':>6} {'Val Acc':>9} {'Val Loss':>9} {'Test Acc':>9} {'Test Loss':>10} {'Seconds':>9}")
    for name, epochs, va, vl, ta, tl, sec in results:
        print(f"{name:<17} {epochs:>6} {va:>9.2%} {vl:>9.4f} {ta:>9.2%} {tl:>10.4f} {sec:>9.1f}")
    best = max(results, key=lambda r: (r[2], -r[3]))
    print(f"Selected by validation: {best[0]}, {best[1]} epochs; test accuracy: {best[4]:.2%}")
    print(f"Saved best-model summary: {best_dir / 'best_model.png'}")
    print(f"Saved comparison: {OUTPUT_DIR / 'comparison'}")
    return results


def main(default_model="VGG16"):
    parser = argparse.ArgumentParser(description="Train one DCNN on BMW/Honda images.")
    parser.add_argument("--model", choices=MODELS, default=default_model)
    parser.add_argument("--epochs", nargs="+", type=int, default=EPOCHS)
    args = parser.parse_args()
    if any(epoch < 1 for epoch in args.epochs):
        parser.error("Epochs must be positive.")
    run_experiments([args.model], args.epochs)


if __name__ == "__main__":
    main()

import argparse
from main import MODELS, EPOCHS, run_experiments


def main():
    parser = argparse.ArgumentParser(description="Run all four models with 5, 10, 20 epochs.")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--epochs", nargs="+", type=int, default=EPOCHS)
    args = parser.parse_args()
    if any(epoch < 1 for epoch in args.epochs):
        parser.error("Epochs must be positive.")
    run_experiments(args.models, args.epochs)


if __name__ == "__main__":
    main()


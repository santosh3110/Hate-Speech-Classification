import argparse
from hate_speech_classifier.pipeline.train_pipeline import run_pipeline
from hate_speech_classifier.pipeline.predict_pipeline import PredictionPipeline

def main():
    parser = argparse.ArgumentParser(description="Hate Speech Classification CLI")
    parser.add_argument("--mode", choices=["train", "predict"], required=True, help="train or predict mode")
    parser.add_argument("--text", type=str, help="Text input for prediction")

    args = parser.parse_args()

    if args.mode == "train":
        print("Running training pipeline...")
        run_pipeline()
        print("Training complete!")

    elif args.mode == "predict":
        if not args.text:
            print("Please provide --text argument for prediction.")
        else:
            print(f"Predicting on: \"{args.text}\"")
            pipe = PredictionPipeline()
            result = pipe.run(args.text)
            print(f"Prediction: {result}")

if __name__ == "__main__":
    main()

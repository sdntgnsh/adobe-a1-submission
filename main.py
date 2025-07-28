# main.py

import os
from predict import predict_all_pdfs

def main():
    """
    Main entry point for the Docker container.
    Reads environment variables for paths and runs the prediction pipeline.
    """
    # These paths are fixed inside the Docker container as per the challenge guidelines
    input_folder = "/app/input"
    output_folder = "/app/output"
    model_path = "model.xgb" # Assumes model is in the same directory

    print("--- Starting Prediction Pipeline ---")
    if not os.path.exists(model_path):
        print(f"CRITICAL ERROR: Model file not found at '{model_path}'")
        return

    predict_all_pdfs(input_folder, model_path, output_folder)
    print("--- Prediction Pipeline Finished ---")


if __name__ == "__main__":
    main()
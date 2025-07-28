# utils.py

import argparse
from rapidfuzz import fuzz

def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train or run predictions for PDF outline extraction.")
    parser.add_argument("--json-folder", default="jsons", help="Folder containing JSON files with labeled outlines.")
    parser.add_argument("--pdf-folder", default="pdfs", help="Folder containing the corresponding PDF files.")
    parser.add_argument("--output-folder", default="output_jsons", help="Folder to save the predicted JSON files.")
    parser.add_argument("--model-path", default="model.xgb", help="Path to save or load the trained model.")
    parser.add_argument("--train", action="store_true", help="Flag to run the training process.")
    return parser.parse_args()

def fuzzy_match_line(text: str, label_items: list[dict], threshold: int = 80) -> dict | None:
    """
    Finds the best fuzzy match for a line of text against a list of labeled items.
    Uses partial ratio to find substrings.
    """
    best_score, best_match = 0, None
    for item in label_items:
        score = fuzz.partial_ratio(text.lower(), item["text"].lower())
        if score > best_score:
            best_score = score
            best_match = item
    return best_match if best_score >= threshold else None

def fuzz_ratio(a: str, b: str) -> float:
    """
    Calculates the similarity ratio between two strings.
    This is a simple wrapper for rapidfuzz.fuzz.ratio.
    """
    return fuzz.ratio(a, b)

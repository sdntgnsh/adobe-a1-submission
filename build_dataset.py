# build_dataset.py

import os
import json
import pandas as pd
from extract_features import extract_pdf_features_in_batches
from utils import fuzzy_match_line, fuzz_ratio
from tqdm import tqdm

def build_training_dataframe(json_folder, pdf_folder):
    """
    Builds the training dataframe with a resilient file-based caching system.
    This version uses the fast, non-semantic feature extractor.
    """
    CACHE_DIR = 'feature_cache_layout_only' # Use a different cache directory
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    all_dataframes = []
    json_files = os.listdir(json_folder)

    print("Building layout-only training dataframe with caching enabled...")
    for fname in tqdm(json_files, desc="Checking Cache & Processing Files"):
        if not fname.endswith(".json"):
            continue

        base = fname.replace(".json", "")
        cache_path = os.path.join(CACHE_DIR, f"{base}.parquet")
        
        # 1. Check if processed data exists in cache
        if os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                all_dataframes.append(df)
                continue
            except Exception as e:
                print(f"Warning: Could not read cache file {cache_path}. Will re-process. Error: {e}")

        # 2. If not cached, process the file
        pdf_path = os.path.join(pdf_folder, f"{base}.pdf")
        if not os.path.exists(pdf_path):
            continue

        with open(os.path.join(json_folder, fname)) as f:
            label_data = json.load(f)

        title = label_data.get("title", "")
        outline = label_data.get("outline", [])
        
        doc_lines = []
        # Call the non-semantic feature extractor
        for pdf_lines_batch in extract_pdf_features_in_batches(pdf_path):
            if not pdf_lines_batch:
                continue
            for line in pdf_lines_batch:
                label = fuzzy_match_line(line["text"], outline)
                line["label"] = label["level"] if label else "O"
                line["is_title"] = int(fuzz_ratio(line["text"].lower(), title.lower()) > 90)
                doc_lines.append(line)
        
        if doc_lines:
            doc_df = pd.DataFrame(doc_lines)
            # 3. Save the newly processed data to its own cache file
            try:
                doc_df.to_parquet(cache_path, engine='pyarrow')
                all_dataframes.append(doc_df)
            except Exception as e:
                print(f"Warning: Could not write to cache file {cache_path}. Error: {e}")

    if not all_dataframes:
        print("Warning: No data was loaded or processed. The final dataframe will be empty.")
        return pd.DataFrame()

    print(f"\nConcatenating data from {len(all_dataframes)} files...")
    final_df = pd.concat(all_dataframes, ignore_index=True)
    print("Dataframe built successfully.")
    return final_df

# predict.py

import os
import json
import pandas as pd
import joblib
from extract_features import extract_pdf_features_in_batches
from tqdm import tqdm

def predict_and_generate_json(pdf_path: str, model, le) -> dict:
    all_predicted_lines = []
    
    for features_batch in extract_pdf_features_in_batches(pdf_path):
        if not features_batch:
            continue
            
        df = pd.DataFrame(features_batch)
        df["is_title"] = 0

        # --- Feature Selection (must match training) ---
        feature_cols = [
            "font_size", "is_bold", "x", "y", "x_norm", "y_norm", "page", "is_title",
            "relative_font_size", "is_all_caps", "is_mostly_digits", "word_count", "char_count",
            "y_gap_from_prev", "x_diff_from_prev", "font_diff_from_prev",
            "ends_with_colon", "starts_with_numbering"
        ]
        
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
                
        X = df[feature_cols]
        
        try:
            preds = model.predict(X)
            df["label"] = le.inverse_transform(preds)
            all_predicted_lines.append(df)
        except Exception as e:
            print(f"CRITICAL ERROR during prediction batch for {pdf_path}: {e}")
            continue

    if not all_predicted_lines:
        return {"title": "", "outline": []}

    final_df = pd.concat(all_predicted_lines, ignore_index=True)

    title = ""
    first_page_df = final_df[final_df['page'] == 1]
    if not first_page_df.empty:
        title_candidates = first_page_df.sort_values(by=["font_size", "y"], ascending=[False, True])
        title = title_candidates.iloc[0]["text"]
    elif not final_df.empty:
        title_candidates = final_df.sort_values(by=["font_size", "page", "y"], ascending=[False, True, True])
        title = title_candidates.iloc[0]["text"]

    outline = []
    for _, row in final_df.iterrows():
        if str(row["label"]).startswith("H"):
            outline.append({"level": row["label"], "text": row["text"], "page": max(int(row["page"]) - 1 , 0)})
            
    return {"title": title, "outline": outline}

def predict_all_pdfs(pdf_folder: str, model_path: str, output_folder: str):
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        model, le = joblib.load(model_path)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Model file not found at {model_path}. Please train the model first.")
        return

    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
    
    print(f"Found {len(pdf_files)} PDFs to predict...")
    for fname in tqdm(pdf_files, desc="Predicting PDFs", unit="file"):
        pdf_path = os.path.join(pdf_folder, fname)
        result = predict_and_generate_json(pdf_path, model, le)
        output_path = os.path.join(output_folder, os.path.splitext(fname)[0] + ".json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n[✓] Prediction complete. Results saved to '{output_folder}'.")

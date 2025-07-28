# extract_features.py

import fitz
import statistics
import unicodedata
import codecs
import re
import os
from tqdm import tqdm
import numpy as np
import gc

# --- Helper functions (dedupe_repeated, etc.) remain the same ---
def dedupe_repeated(text: str) -> str:
    return re.sub(r'\b(\w{3,})\b(?:\s+\1\b)+', r'\1', text, flags=re.IGNORECASE).strip()

def unescape_string(s: str) -> str:
    try: s = codecs.decode(s, "unicode_escape")
    except Exception: pass
    try: s = s.encode("latin1").decode("utf-8")
    except Exception: pass
    return s

def normalize_text(text: str) -> str:
    replacements = {"\u2018": "'", "\u2019": "'", "\u201C": '"', "\u201D": '"', "\u2013": "-", "\u2014": "-", "\u00A0": " "}
    for k, v in replacements.items(): text = text.replace(k, v)
    nfkd = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", stripped)


def extract_pdf_features_in_batches(pdf_path: str):
    """
    Processes a PDF page-by-page, extracting only layout and pattern features.
    This version is optimized for speed and does not perform semantic analysis.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF {pdf_path}: {e}")
        return

    PAGE_LIMIT = 60
    
    for page_num in range(min(doc.page_count, PAGE_LIMIT)):
        page_lines = []
        page = doc.load_page(page_num)
        
        width, height = page.rect.width, page.rect.height
        if width == 0 or height == 0: continue
        
        blocks = page.get_text("dict")["blocks"]
        words = page.get_text("words")
        if not words: continue

        blocks_map = {b.get("number"): b for b in blocks}
        
        all_font_sizes = [s["size"] for b in blocks for l in b.get("lines", []) for s in l.get("spans", [])]
        avg_page_font_size = statistics.mean(all_font_sizes) if all_font_sizes else 0
        median_h = statistics.median([w[3] - w[1] for w in words]) if words else 10
        
        line_groups = []
        for w in sorted(words, key=lambda w: w[1]):
            y0 = w[1]; placed = False
            for grp in line_groups:
                if abs(y0 - grp["y0_vals"][0]) < median_h * 0.5:
                    grp["words"].append(w); grp["y0_vals"].append(y0); placed = True; break
            if not placed: line_groups.append({"words": [w], "y0_vals": [y0]})
            
        for grp in line_groups:
            try:
                sorted_w = sorted(grp["words"], key=lambda w: w[0])
                raw = " ".join(w[4].strip() for w in sorted_w)
                text = dedupe_repeated(normalize_text(unescape_string(raw))).strip()
                if not text: continue

                b_num, l_num = sorted_w[0][5], sorted_w[0][6]
                spans = []
                target_block = blocks_map.get(b_num)
                if target_block and l_num < len(target_block.get("lines", [])):
                    spans = target_block["lines"][l_num].get("spans", [])

                font_sizes = [s["size"] for s in spans]
                font_size = statistics.mean(font_sizes) if font_sizes else 0
                is_bold = 1 if sum("bold" in s["font"].lower() for s in spans) > len(spans) / 2 else 0
                
                page_lines.append({
                    "text": text, "page": page_num + 1, "font_size": font_size, "is_bold": is_bold,
                    "x": sorted_w[0][0], "y": grp["y0_vals"][0], "y1": sorted_w[-1][3],
                    "x_norm": (sorted_w[0][0] / width) if width > 0 else 0, "y_norm": (grp["y0_vals"][0] / height) if height > 0 else 0,
                    "relative_font_size": font_size / avg_page_font_size if avg_page_font_size > 0 else 0,
                    "ends_with_colon": 1 if text.strip().endswith(':') else 0,
                    "starts_with_numbering": 1 if re.match(r'^((\d{1,2}(\.\d*)*\s)|([A-Za-z]\.)|(\([a-z\d]\))|(Phase\s[IVXLCDM]+)|(Section\s\d+))', text.strip(), re.IGNORECASE) else 0,
                    "word_count": len(text.split()), "char_count": len(text),
                    "is_all_caps": 1 if all(c.isupper() for c in text if c.isalpha()) else 0,
                    "is_mostly_digits": 1 if text.strip() and sum(c.isdigit() for c in text) / len(text.strip()) > 0.6 else 0
                })
            except Exception: continue
        
        if not page_lines: continue

        contextualized_page = []
        for i, line in enumerate(page_lines):
            if i > 0:
                prev_line = page_lines[i-1]
                line["y_gap_from_prev"] = line["y"] - prev_line["y1"]
                line["x_diff_from_prev"] = line["x"] - prev_line["x"]
                line["font_diff_from_prev"] = line["font_size"] - prev_line["font_size"]
            else:
                line["y_gap_from_prev"], line["x_diff_from_prev"], line["font_diff_from_prev"] = 0, 0, 0
            contextualized_page.append(line)
        
        yield contextualized_page
        
        del page_lines, contextualized_page
        gc.collect()

    doc.close()

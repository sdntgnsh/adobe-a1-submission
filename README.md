# PDF Outline Extraction - Adobe Hackathon 2025

This project is a solution for Challenge 1a, designed to extract structured outline data from PDF documents and output it into a standardized JSON format. The entire pipeline is containerized with Docker and optimized to meet the strict performance and resource constraints of the competition.

## Solution Architecture

The core of this solution is a machine learning model trained to classify each line of text within a PDF. It determines whether a line is a heading (H1, H2, etc.) or regular body text (O).

### 1. Model

The model is an **XGBoost Classifier**, a highly efficient and powerful gradient boosting library. It was chosen for its exceptional performance on tabular data and its ability to run quickly on a CPU. The final trained model (`model.xgb`) is included in the repository and is well under the 200MB size constraint.

### 2. Feature Engineering

To meet the strict **sub-10-second execution time**, this solution deliberately avoids slow semantic analysis. Instead, it relies on a rich set of fast-to-compute layout, pattern, and contextual features. The key features include:

* **Layout Features:**
    * Font size and boldness.
    * `relative_font_size`: The line's font size compared to the average on the page.
    * `x_norm`, `y_norm`: The normalized X/Y coordinates of the line.
* **Pattern-Based Features:**
    * `is_all_caps`: Whether the line is in all uppercase.
    * `ends_with_colon`: A strong indicator of a leading title.
    * `starts_with_numbering`: Detects patterns like "1.1", "A.", or "Phase I".
    * `word_count` and `char_count`.
* **Contextual Features:**
    * `y_gap_from_prev`: The vertical space between the current line and the previous one.
    * `font_diff_from_prev`: The change in font size from the previous line.

This feature-rich, non-semantic approach allows the model to make highly accurate predictions based on the visual and structural cues within the document, ensuring the performance requirements are met.

## Dataset Generation using LLMs

The training data for the model was generated synthetically using a Large Language Model (LLM) to create a high-quality labeled dataset without manual annotation. This process allowed for the rapid creation of diverse and accurate training examples.

The workflow was as follows:

1.  **Text Extraction:** For a given PDF, the raw text content of each page was extracted.
2.  **LLM Prompting:** A powerful LLM (such as a model from the GPT or Gemini family) was prompted with the extracted text. The prompt was carefully engineered to ask the model to act as a document analysis expert.
3.  **Structured JSON Output:** The core of the prompt instructed the LLM to return its analysis in a specific JSON format. A simplified version of the prompt looked like this:
    > "Given the following text extracted from a PDF document, please analyze its structure and generate a JSON object. The JSON object must contain two keys: a 'title' for the document, and an 'outline' which is an array of objects. Each object in the 'outline' array should represent a heading and contain three keys: 'text' (the heading text), 'level' (the heading level, e.g., 'H1', 'H2'), and 'page' (the page number where the heading appeared)."
4.  **Creating the Ground Truth:** The JSON file returned by the LLM was saved as the "ground truth" label for that PDF. This process was repeated for a diverse set of documents to build the `jsons` folder.
5.  **Model Training:** The `build_dataset.py` script then used these LLM-generated JSON files to assign the correct labels to the features extracted from the corresponding PDFs, creating the final training dataset for the XGBoost model.

This semi-automated approach enabled the creation of a large, high-quality dataset that captures a wide variety of document structures.

## How to Build and Run

The solution is fully containerized and expects to be run in an environment with no internet access.

### Build Command

To build the Docker image, navigate to the root of this repository and run:

ocker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .


### Run Command

To run the solution, ensure you have an `input` directory with your test PDFs and an empty `output` directory. From the directory containing your `input` and `output` folders, run:

docker run --rm -v $(pwd)/input:/app/input:ro -v $(pwd)/output:/app/output --network none mysolutionname:somerandomidentifier

The container will automatically process all PDFs in `/app/input` and place the resulting JSON files in `/app/output`.

## Project Structure

The repository contains only the essential files for the prediction pipeline:

.
├── model.xgb                  # The trained XGBoost model.
├── main.py                    # Main entry point for the container.
├── predict.py                 # Core prediction logic.
├── extract_features.py        # High-speed feature extraction script.
├── utils.py                   # Utility functions.
├── Dockerfile                 # Docker container configuration.
├── requirements.txt           # Required Python libraries for prediction.
└── README.md                  # This documentation file.

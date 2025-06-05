# Multimodal Dataset Filtering Pipeline
This repository contains a Python-based pipeline for filtering multimodal datasets, designed to process datasets containing text, images, questions, and answers. The pipeline applies a series of customizable filters to clean and refine the dataset, ensuring high-quality data for downstream tasks such as visual document question answering. It supports Hugging Face DatasetDict datasets and includes visualization tools for analyzing the filtering process.

Features:

- Modular Filters: Apply filters such as text length, n-gram overlap, language detection, and question deduplication.
- Multimodal Support: Handles datasets with text, OCR text, images, and question-answer pairs.
- Visualization: Generates histograms, summary tables, and visualizations of filtered datasets.
- Intermediate Saving: Options to save all or subsampled intermediate datasets for debugging and analysis.
- Performance Tracking: Measures and reports the number of samples/questions remaining and filter processing times.

## Requirements

Python 3.9+

## Installation

Clone the repository:
```
git clone https://github.com/snova-jonathanl/HuDocVQA.git
cd HuDocVQA
```

Install dependencies:
```
pip install -r requirements.txt
```

Ensure you have a valid SambaNova Cloud API key for the DedupQuestionsFilter.

## Usage
The pipeline is run via a command-line interface. Below is an example command to run the pipeline:
```
python pipeline.py \
  --dataset_path /path/to/your/dataset \
  --output_dir /path/to/output \
  --sn_api_key your_sambanova_api_key \
  --sn_model Meta-Llama-3.1-8B-Instruct \
  --save_intermediate subsample \
  --subsample_size 10
```

## Arguments
- `--dataset_path`: Path to the input Hugging Face dataset directory.
- `--output_dir`: Directory to save the filtered dataset and visualizations.
- `--sn_api_key`: SambaNova Cloud API key for the deduplication filter.
- `--sn_model`: SambaNova model name (default: `Meta-Llama-3.1-8B-Instruct`).
- `--save_intermediate`: Save intermediate datasets (`none`, `subsample`, or `all`). Default: `subsample`.
- `--subsample_size`: Number of samples to save when using subsample mode. Default: `10`.

## Output

- Filtered Dataset: Saved as a Hugging Face dataset in `output_dir/final_dataset`.
- Intermediate Datasets: Optionally saved in `output_dir/intermediate` (all or subsampled).
- Histograms for each filter (`intermediate/<filter_name>_<index>_histogram.png`).
- Filtering summary table (`output_dir/filtering_summary_table.png`).
- Plot of remaining samples after each filter (`output_dir/filtering_process.png`).
- Subsampled dataset visualizations (if `--save_intermediate=subsample`).

## Filters
The pipeline uses a modular filter system, with each filter inheriting from a Filter base class. Each filter:

1. Processes examples and decides whether to keep or remove them.
2. Updates question-answer pairs to maintain consistency.
3. Tracks metrics for visualization (e.g., histograms of filter values).

Available Filters

- TextLengthFilterOCR: Ensures text or OCR content meets a minimum length threshold.
- NGramFilter: Filters based on character-level n-gram overlap between text and questions.
- LanguageFilter: Ensures text and questions match a specified language (e.g., Hungarian).
- DedupQuestionsFilter: Removes duplicate or paraphrased questions using a language model.

## Directory Structure
```
data_cleaning/
├── filters/
│   ├── base_filter.py
│   ├── text_length_filter.py
│   ├── n_gram_overlap_filter.py
│   ├── lang_filter.py
│   ├── lang_filter_ocr.py
│   ├── deduplicate_questions_filter.py
├── pipeline.py
├── requirements.txt
├── README.md
```

## Notes

The [googletrans](https://pypi.org/project/googletrans/) library is used for translation in visualizations but may encounter rate limits. Consider using a more robust translation API for production use.

The DedupQuestionsFilter requires a valid SambaNova API key, which you can sign up for [here](https://cloud.sambanova.ai/dashboard). Ensure the API is accessible and the key is valid.

The pipeline assumes the input dataset is a HuggingFace Dataset that was saved to disk (using `dataset.save_to_disk`). We assume the dataset has fields like `text`, `ocr`, `questions`, `answers`, and `image`. Ensure your dataset matches this structure.

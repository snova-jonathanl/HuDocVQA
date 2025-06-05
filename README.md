# Synthetic Document Question Answering in Hungarian
<p align="center">
    📖 <a href="https://arxiv.org/abs/2505.23008" target="_blank">Paper</a> • 🤗 <a href="https://huggingface.co/datasets/jlli/HuDocVQA" target="_blank">HuDocVQA</a> • 🤗 <a href="https://huggingface.co/datasets/jlli/HuDocVQA-manual" target="_blank">HuDocVQA-manual</a> • 🤗 <a href="https://huggingface.co/datasets/jlli/HuCCPDF" target="_blank">HuCCPDF</a>
</p>
Codebase for reproducing HuDocVQA and HuCCPDF, two datasets for benchmarking and training LLMs for visual document question answering in Hungarian. Check out our datasets on HuggingFace!

## HuCCPDF
Coming soon.

## HuDocVQA

### Synthetic QA Generation
Coming soon.

### Filtering Synthetic Questions & Answers
Assuming the result of synthetic QA generation was written to disk as a HuggingFace dataset:
```
$ cd HuDocVQA/data_cleaning
$ python pipeline.py \
  --dataset_path /path/to/your/dataset \
  --output_dir /path/to/output \
  --sn_api_key your_sambanova_api_key \
  --sn_model Meta-Llama-3.1-8B-Instruct \
  --save_intermediate subsample \
  --subsample_size 10
```
See [data_cleaning/README.md](https://github.com/snova-jonathanl/HuDocVQA/blob/main/data_cleaning/README.md) for more detailed installation instructions.

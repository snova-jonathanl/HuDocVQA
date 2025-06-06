# Synthetic QA Generation
We show how to generate synthetic questions and answers from ground-truth text.

## Requirements
Python 3.9+
Run `pip install -r requirements.txt`.
You will need to clone the Tesseract [tessdata](https://github.com/tesseract-ocr/tessdata) repository for running OCR on document images.
```
$ pwd
<tessdata_prefix>
$ git clone https://github.com/tesseract-ocr/tessdata.git
```
## Invocation
Assume your dataset is saved to disk as a HuggingFace dataset with the following structure:
```
$ ls path/to/input/dataset
dataset_dict.json  test  train  val
$ python3
>>> from datasets import load_from_disk
>>> ds = load_from_disk('path/to/input/dataset')
>>> ds
DatasetDict({
    train: Dataset({
        features: ['image', 'text', 'ocr'],
        num_rows: 50000
    })
    test: Dataset({
        features: ['image', 'text', 'ocr'],
        num_rows: 1000
    })
    val: Dataset({
        features: ['image', 'text', 'ocr'],
        num_rows: 1000
    })
}
```
Where each `'image'` feature is a PIL image, `'text'` is a string corresponding to the text in the associated image. `'ocr'` is an optional field that is the result of running OCR on the image to obtain an additional source of ground truth text.
Then, run the following:
```
export OPENAI_API_KEY=<sambanova_cloud_api_key>
export TESSDATA_PREFIX=<tessdata_prefix>/tessdata # path to the tessdata repository you cloned earlier
python generate_synqa.py \
       --input-dir path/to/input/dataset \
       --output-dir path/to/output/dataset
```
You should obtain a dataset with the following additional new fields:
```
$ python3
>>> from datasets import load_from_disk
>>> ds = load_from_disk('path/to/output/dataset')
>>> ds
DatasetDict({
    train: Dataset({
        features: ['image', 'text', 'ocr', 'questions', 'answers'],
        num_rows: 50000
    })
    test: Dataset({
        features: ['image', 'text', 'ocr', 'questions', 'answers'],
        num_rows: 1000
    })
    val: Dataset({
        features: ['image', 'text', 'ocr', 'questions', 'answers'],
        num_rows: 1000
    })
}
```
## Future Improvements
The current script is single-threaded and iterates sequentially through the dataset. Releasing batches of API requests to avoid rate limits may speed up generation time.

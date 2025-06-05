# filters/text_length_filter.py
from .base_filter import Filter

class TextLengthFilterOCR(Filter):
    def __init__(self, min_length: int):
        super().__init__()
        self.min_length = min_length

    def add_to_histogram(self, length):
        self.hist_list.append(length)

    def filter_example(self, example):
        #filter it out if both the OCR or actual text length is too short
        value = max(len(example["text"]), len(example["ocr"]))
        return example, value, value < self.min_length

    def __str__(self):
        return f"TextLengthFilterOCR(min_length={self.min_length})"


from langdetect import detect
from .base_filter import Filter

class LanguageFilterOCR(Filter):
    def __init__(self, language: str):
        """
        Initialize the LanguageFilter with the expected language code.
        :param language: The language code to filter by (e.g., 'en', 'fr').
        """
        super().__init__()
        self.language = language

    def add_to_histogram(self, langs):
        for lang in langs:
            if lang not in self.hist_counts:
                self.hist_counts[lang] = 1
            else:
                self.hist_counts[lang] += 1
        

    def filter_example(self, example):
        """
        Filter an example by checking if the detected languages of the text, question, and other fields match the expected language.
        Also ensures that answers are filtered to match the updated questions.
        :param example: A dictionary containing 'text', 'question', or other relevant keys.
        :return: The updated example, detected languages, and a flag indicating removal.
        """
        # Detect languages for the relevant fields
        try:
            text_language = detect(example.get("text", ""))
        except:
            text_language = "None"
        try:
            ocr_lang = detect(example.get("ocr"))
        except:
            ocr_lang = "None"

        langs = [text_language, ocr_lang]
        remove = text_language != self.language and ocr_lang != self.language 

        # Filter questions based on detected language
        orig_questions = example.get("questions", [])
        orig_answers = example.get("answers", [])
        filtered_questions = [question for question in orig_questions if detect(question) == self.language]

        # Use the helper function to filter answers based on the updated questions
        filtered_answers = self.filter_answers(orig_questions, orig_answers, filtered_questions)

        # Update the example
        example['questions'] = filtered_questions
        example['answers'] = filtered_answers

        # Determine if the example should be removed
        remove = remove or len(filtered_questions) == 0

        return example, langs, remove
        

    def __str__(self):
        """
        Return a string representation of the filter.
        """
        return f"LanguageFilterOCR(language={self.language})"


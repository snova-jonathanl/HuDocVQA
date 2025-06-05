# filters/text_length_filter.py
from .base_filter import Filter

class NGramFilter(Filter):
    def __init__(self, n_gram: int, threshold: float):
        super().__init__()
        self.n_gram = n_gram
        self.threshold = threshold

    def generate_ngrams(self, text, n):
        """Helper function to generate character-level n-grams from a given text."""
        # Remove spaces if you want pure character n-grams (optional)
        text = text.replace(" ", "")
        return [text[i:i+n] for i in range(len(text) - n + 1)]

    def add_to_histogram(self, overlap_percentages):
        self.hist_list += overlap_percentages
    

    def get_filter_value(self, example, question):
        """
        Converts the data point to a value that can be used to deterime if it should be filtered out
        """
        text = example["text"]
        if len(text) < self.n_gram or len(question) < self.n_gram:
            return False  # Do not filter examples where n-grams cannot be generated

        # Calculate overlap of n-grams
        ngrams_text = self.generate_ngrams(text, self.n_gram)
        ngrams_question = self.generate_ngrams(question, self.n_gram)
        
        # Convert n-grams to sets for efficient computation
        set_ngrams_text = set(ngrams_text)
        set_ngrams_question = set(ngrams_question)

        # Compute the intersection of the two sets of n-grams
        overlap = len(set_ngrams_text & set_ngrams_question)
        
        # Calculate the percentage overlap
        overlap_percentage = overlap / len(set_ngrams_question)
        
        # Return the result based on the threshold
        return overlap_percentage

    def filter_example(self, example):
        """Filter example based on n-gram overlap between text and questions."""
        # Initialize lists for values and questions
        values = []
        questions = []
        orig_questions = example['questions']
        orig_answers = example.get('answers', [])

        # Iterate over the original questions to compute overlap and filter questions
        for question in orig_questions:
            overlap_percentage = self.get_filter_value(example, question)
            values.append(overlap_percentage)
            if overlap_percentage > self.threshold:
                questions.append(question)

        # Use the helper function to filter answers based on the filtered questions
        filtered_answers = self.filter_answers(orig_questions, orig_answers, questions)

        # Update the example with filtered questions and answers
        example['questions'] = questions
        example['answers'] = filtered_answers

        return example, values, len(questions) == 0

    def __str__(self):
        return f"NGramFilter(n_gram={self.n_gram}, threshold={self.threshold})"


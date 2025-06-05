# filters/text_length_filter.py
from .base_filter import Filter
from openai import OpenAI
import ast 
import time

class DedupQuestionsFilter(Filter):
    def __init__(self, model: str, api_key: str):
        super().__init__()
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.sambanova.ai/v1",
        )
        self.model = model 
        self.api_key = api_key

    def add_to_histogram(self, value):
        self.hist_list.append(value)
        
    def extract_list_from_string(self, input_string):
        """
        Extracts and returns a Python list from a string containing a list within square brackets.

        Args:
            input_string (str): The input string containing the list.

        Returns:
            list: The extracted Python list, or None if no valid list is found.
        """
        try:
            # Find the part of the string within square brackets
            start = input_string.find('[')
            end = input_string.rfind(']')
            if start == -1 or end == -1 or start > end:
                return None  # No valid list found
            
            # Extract and safely evaluate the content
            list_content = input_string[start:end + 1]
            return ast.literal_eval(list_content)
        except (ValueError, SyntaxError):
            return None  # Return None if there's a syntax issue

    def query_llm(self, prompt, rec=0):
        time.sleep(2)
        if rec == 20:
            raise ValueError("Too many API fails")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=1024,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            if response is None or response.choices is None:
                print(f"[WARN] Empty response or missing choices. Retrying ({rec+1}/20)...")
                time.sleep(10)
                return self.query_llm(prompt, rec+1)
        except Exception as e:
            print(f"[ERROR] Exception during API call (attempt {rec+1}/20): {e.__class__.__name__}: {e}")
            time.sleep(10)
            return self.query_llm(prompt, rec+1)
        # Get the completion response
        completion = response.choices[0].message.content
        return completion

    def deduplicate_list(self, questions):
        if len(questions) <= 1:
            return questions
        prompt = f"Here is a list of questions, please deduplicate the list so that any if any pair of questions are the same, similar or paraphrases then remove them, and only return the deduplicated list. Please reply with a list of the deduplicated questions that is python list format, with no other text or code, just the python list. The deduplicated list must be a subset of original list, of less than or equal length. \nOriginal Question List: {str(questions)}\nDeduplicated Question List: "
        response = self.query_llm(prompt)
        dedup_list = self.extract_list_from_string(response)
        if dedup_list is None or len(dedup_list) > len(questions):
            return questions
        return dedup_list

    def filter_example(self, example):
        """Filter example based on n-gram overlap between text and questions."""
        # Store original questions and answers
        orig_questions = example['questions']
        orig_answers = example['answers']

        # Deduplicate questions and get the updated list
        dedup_questions = self.deduplicate_list(orig_questions)
        final_questions = []
        for question in dedup_questions:
            if question.strip() in orig_questions:
                final_questions.append(question.strip())

        if final_questions is None:
            return example, 0, False

        # Use the helper function to filter answers based on the new questions
        answers = self.filter_answers(orig_questions, orig_answers, final_questions)
        # Update the example
        example['questions'] = final_questions
        example['answers'] = answers

        # Determine if the example should be removed
        remove = len(final_questions) == 0
        value = len(orig_questions) - len(final_questions)

        return example, value, remove

    def __str__(self):
        return f"DedupQuestionsFilter(model={self.model})"


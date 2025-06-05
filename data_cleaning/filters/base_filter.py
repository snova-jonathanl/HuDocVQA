# filters/base_filter.py
from abc import ABC, abstractmethod
from datasets import Dataset
import os
import numpy as np
import matplotlib.pyplot as plt

class Filter(ABC):
    def __init__(self):
        """
        Initialize the ResolutionFilter with the minimum resolution requirements.
    :param min_width: The minimum width for the image.
        :param min_height: The minimum height for the image.
        """
        self.hist_list = []
        self.hist_counts = {}

    @abstractmethod
    def add_to_histogram(self, vals):
        raise NotImplementedError("add_to_histogram is not implemented yet.")

    def plot_histogram(self, output_figure_path):
        """
        Plot a histogram based on self.hist_list or self.hist_counts.

        If self.hist_list is defined, plots a histogram of the distribution of the values
        (y-axis as likelihood and x-axis as value). Outliers are excluded for better visualization.
        If self.hist_counts is defined, plots a bar chart with keys as x-axis and counts as y-axis.

        The plot is saved to the specified output_figure_path.

        :param output_figure_path: Path to save the figure.
        """
        if len(self.hist_list) > 0:
            # Plot histogram of self.hist_list
            values = np.array(self.hist_list)

            # Exclude outliers (e.g., values beyond 1.5 * IQR)
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            filtered_values = values[(values >= lower_bound) & (values <= upper_bound)]

            # Plot normalized histogram
            plt.figure(figsize=(8, 6))
            plt.hist(filtered_values, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
            plt.xlabel('Value')
            plt.ylabel('Likelihood')
            plt.title('Histogram of Values')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
        elif len(self.hist_counts) > 0:
            # Plot bar chart of self.hist_counts
            keys = list(self.hist_counts.keys())
            counts = list(self.hist_counts.values())

            plt.figure(figsize=(8, 6))
            plt.bar(keys, counts, color='skyblue', edgecolor='black', alpha=0.7)
            plt.xlabel('Categories')
            plt.ylabel('Counts')
            plt.title('Histogram of Counts')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
        else:
            print("NO entries in hist counts or hist list")
            a = 1

        # Save the figure to the specified path
        plt.tight_layout()
        plt.savefig(output_figure_path)
        plt.close()

    def __str__(self):
        """
        Returns a string representation of the filter.
        """
        params = ", ".join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"{self.__class__.__name__}({params})"

    @abstractmethod
    def filter_example(self, example):
        """
        Return the updated version of the example, and if it should be filtered out. 
        """
        pass


    def apply(self, dataset: Dataset) -> (Dataset, Dataset):
        """
        Applies the filter to the dataset and returns the filtered and remaining datasets.
        """
        print(f"Filtering with {str(self)}")
        
        # Create a map function to add a 'split' key
        def split_function(example):
            # iterate through the questions if the filter requires it
            updated_example, hist_values, removed = self.filter_example(example)
            self.add_to_histogram(hist_values)
            updated_example['filter_value'] = hist_values
            if removed:  # Example should be removed
                updated_example['split'] = 'removed'
            else:  # Example should be kept
                updated_example['split'] = 'remaining'

            return updated_example
           
        # Apply the map function
        dataset_split = dataset.map(split_function, load_from_cache_file=False)
        
        # Filter out the 'removed' and 'remaining' groups
        removed_group = dataset_split.filter(lambda x: x['split'] == 'removed')
        remaining_group = dataset_split.filter(lambda x: x['split'] == 'remaining')

        return remaining_group, removed_group

    def filter_answers(self, orig_questions, orig_answers, new_questions):
        """
        Filters the original answers to match the indices of the new questions.
        
        :param orig_questions: List of original questions.
        :param orig_answers: List of original answers corresponding to the original questions.
        :param new_questions: List of filtered questions.
        :return: List of filtered answers corresponding to the new questions.
        """
        # Create a mapping of original question indices to their answers
        question_to_answer = {
            question: orig_answers[idx] for idx, question in enumerate(orig_questions) if idx < len(orig_answers)
        }
        
        # Filter the answers based on the new questions
        new_answers = [question_to_answer[question] for question in new_questions if question in question_to_answer]
        
        return new_answers

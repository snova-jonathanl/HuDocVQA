from datasets import load_from_disk, Dataset, concatenate_datasets, DatasetDict
from typing import List
import os
import re
import shutil
import matplotlib.pyplot as plt
from filters.n_gram_overlap_filter import NGramFilter
from filters.lang_filter import LanguageFilter
from filters.base_filter import Filter
from filters.text_length_filter_with_ocr import TextLengthFilterOCR
from filters.lang_filter_ocr import LanguageFilterOCR
from filters.deduplicate_questions_filter import DedupQuestionsFilter
import time
from googletrans import Translator
import asyncio
import argparse


def check_questions_answers_length_match(dataset: DatasetDict) -> None:
    """
    Check if the length of 'questions' matches the length of 'answers' for each sample in the dataset.
    Prints a warning for any sample where the lengths do not match.

    Args:
        dataset (DatasetDict): A Hugging Face DatasetDict containing dataset splits.
    """
    mismatches_found = False
    for split in dataset:
        if split in ['train', 'test', 'val'] and len(dataset[split]) > 0:
            for idx, sample in enumerate(dataset[split]):
                try:
                    # Check if both 'questions' and 'answers' keys exist
                    if 'questions' in sample and 'answers' in sample:
                        questions_len = len(sample['questions'])
                        answers_len = len(sample['answers'])
                        if questions_len != answers_len:
                            mismatches_found = True
                            print(
                                f"Warning: Mismatch in split '{split}', sample index {idx}: "
                                f"questions length ({questions_len}) does not match "
                                f"answers length ({answers_len})"
                            )
                    else:
                        print(
                            f"Warning: Missing 'questions' or 'answers' in split '{split}', sample index {idx}"
                        )
                except Exception as e:
                    print(
                        f"Error processing split '{split}', sample index {idx}: {e}"
                    )
    
    if not mismatches_found:
        print("All samples have matching questions and answers lengths.")

class DatasetFilteringPipeline:
    def __init__(self, filters: List[Filter], output_dir: str, save_intermediate: str = "none", subsample_size: int = 10):
        """
        Initialize the pipeline.

        Args:
            filters (List[Filter]): A list of filters to apply.
            output_dir (str): The directory where the output will be saved.
            save_intermediate (str, optional): How to save intermediate datasets. Options are "all", "subsample", and "none". Defaults to "none".
            subsample_size (int, optional): The number of examples to save when using subsampling. Defaults to 10.
        """
        self.filters = filters
        self.output_dir = output_dir
        self.save_intermediate = save_intermediate
        self.subsample_size = subsample_size
        self.translator = Translator()

    def _create_intermediate_dir(self):
        intermediate_dir = os.path.join(self.output_dir, "intermediate")
        os.makedirs(intermediate_dir, exist_ok=True)
        return intermediate_dir

    def _get_total_length(self, dataset):
        return sum(len(dataset[split]) for split in ['train', 'test', 'val'] if split in dataset)

    def _get_total_questions(self, dataset):
        return sum(
            sum(len(row['questions']) for row in dataset[split]) 
            for split in ['train', 'test', 'val'] 
            if split in dataset
        )
    

    def _split_has_data(self, split, dataset):
        return split in dataset and len(dataset[split]) > 0

    def _all_splits_have_data(self, dataset):
        for split in ['train', 'test', 'val']:
            if not self._split_has_data(split, dataset):
                print(f"Warning: {split} split is missing or has no data.")
                return False
        return True

    def _subsample_dataset(self, dataset, size):
        """
        Subsample the dataset.

        Args:
            dataset (DatasetDict): A Hugging Face DatasetDict object containing dataset splits (e.g., {"train": train_dataset, "test": test_dataset}).
            size (int): The number of examples to keep.

        Returns:
            Dataset: The subsampled dataset as a single Hugging Face Dataset.
        """
        subsampled_splits = {}
        for split in dataset:
            # Ensure we don't try to select more samples than exist in the split
            actual_size = min(size, len(dataset[split]))
            if len(dataset[split]) > 0:
                subsampled_splits[split] = dataset[split].shuffle().select(range(actual_size))
            
        # Concatenate all splits into a single Dataset
        return DatasetDict(subsampled_splits)

    async def _visualize_multimodal_dataset(self, dataset, path, max_samples=None):
        """
        Visualize a multimodal dataset by saving images with their corresponding questions and text.
        Args:
            dataset (dict): A dictionary of dataset splits.
            path (str): The path to save the visualization image.
            max_samples (int, optional): Maximum number of samples to visualize. 
                                        If None, uses a default limit of 20 samples.
        """
        # Collect samples
        samples = []
        for split in dataset.keys():
            for data_pt in dataset[split]:
                if len(samples) < (max_samples or 20):
                    samples.append(data_pt)
                else:
                    break
            if len(samples) >= (max_samples or 20):
                break
        
        num_samples = len(samples)
        
        # If no samples, create an empty figure
        if num_samples == 0:
            plt.figure(figsize=(8, 4))
            plt.text(0.5, 0.5, 'No samples to visualize', ha='center', va='center')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(path, bbox_inches='tight')
            plt.close()
            print(f"Empty dataset visualization saved to {path}")
            return
        
        # Limit the number of samples to prevent excessive memory usage
        max_display_samples = 20
        samples = samples[:max_display_samples]
        num_samples = len(samples)
        
        # Create a figure with controlled height
        plt.figure(figsize=(10, min(10 * num_samples, 100)))
        
        # Create a vertical layout
        for i, sample in enumerate(samples):
            try:
                plt.subplot(num_samples, 1, i + 1)
                
                # Check if the sample has the expected keys
                if all(key in sample for key in ['image', 'questions', 'text']):
                    plt.imshow(sample['image'])
                    
                    # Truncate text for display
                    question = sample['questions'][0][:100]
                    text = sample['text'][:100]
                    translated_question = await self.translator.translate(question)
                    question_english = translated_question.text

                    translated_text = await self.translator.translate(text)
                    text_english = translated_text.text
                    value = sample['filter_value']
                    if isinstance(value, float):
                        value = round(value, 4)
                    title = (
                        f"--------------------------------------------------------------------------\n"
                        f"Sample {i+1}\n"
                        f"Filter value: {value}\n"
                        f"Question (Original):\n{question}{'...' if len(question) > 100 else ''}\n\n"
                        f"Question (English):\n{question_english}{'...' if len(question_english) > 100 else ''}\n\n"
                        f"Text (Original):\n{text}{'...' if len(text) > 200 else ''}\n\n"
                        f"Text (English):\n{text_english}{'...' if len(text_english) > 200 else ''}\n\n"
                    )
                    title = title.replace('$', "[DOLLAR]")
                    plt.title(
                        title,
                        fontsize=14, 
                        wrap=True
                    )
                else:
                    plt.text(0.5, 0.5, 'Invalid sample', ha='center', va='center')
                
                plt.axis('off')
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
        
        # Adjust layout and save
        plt.tight_layout()
        
        try:
            plt.savefig(path, bbox_inches='tight', dpi=300)
        except ValueError as ve:
            # Fallback: reduce DPI if image is still too large
            print(f"Warning: Reduced DPI due to large image size. Original error: {ve}")
            plt.savefig(path, bbox_inches='tight', dpi=100)
        
        plt.close()  # Close the figure to free up memory
        print(f"Dataset visualization saved to {path}")

    def _plot_remaining_samples(self, num_samples, original_length, filter_names):
        filter_names = list(map(self.insert_newline_before_parenthesis, filter_names))
        percentages = [num / original_length * 100 for num in num_samples]
        plt.figure(figsize=(10, 6))
        plt.plot(percentages, marker='o')
        plt.xticks(range(len(filter_names)), filter_names, rotation=90)
        plt.xlabel('Filter')
        plt.ylabel('Remaining percentage of samples')
        plt.ylim(0, 100)
        plt.title('Remaining samples after each filter')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'filtering_process.png'))

    def insert_newline_before_parenthesis(self, s):
        s = re.sub(r'\(', '\n(', s)
        return re.sub(r',', ',\n', s)

    def _create_summary_table(self, filter_names, remaining_samples, removed_samples, filter_times_per_sample, questions_remaining, questions_removed):
        # Dynamically adjust figure size based on the number of filters
        num_filters = len(filter_names)
        fig, ax = plt.subplots(figsize=(min(12, num_filters * 2), 6))  # Adjust width dynamically
        ax.axis('tight')
        ax.axis('off')
        
        # Format filter names to handle line breaks
        filter_names = list(map(self.insert_newline_before_parenthesis, filter_names))
        
        # Prepare table data
        table_data = [
            ["Stage"] + filter_names,
            ["Remaining Examples"] + [str(x) for x in remaining_samples],
            ["Removed Examples"] + [str(x) for x in removed_samples],
            ["Remaining Questions"] + [str(x) for x in questions_remaining],
            ["Removed Questions"] + [str(x) for x in questions_removed],
            ["Filter Time per\n1000 Samples (s)"] + filter_times_per_sample
        ]
        
        # Create and style the table
        table = ax.table(
            cellText=table_data,
            cellLoc='center',  # Center-align text
            loc='center',
            colWidths=[0.2] + [max(0.1, 0.8 / num_filters)] * num_filters  # Adjust column widths dynamically
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(filter_names) + 1)))
        
        # Adjust row heights to accommodate multi-line text
        cell_dict = table.get_celld()
        for (row, col), cell in cell_dict.items():
            cell.set_height(0.15)  # Set row height to accommodate multi-line text
            if row == 0:  # Header row
                cell.set_text_props(weight='bold')  # Bold headers
                cell.set_facecolor('#f2f2f2')  # Light gray background for headers
            cell.set_edgecolor('#d0d0d0')  # Add light gray gridlines for better readability
        
        # Save the table as an image
        output_path = os.path.join(self.output_dir, 'filtering_summary_table.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)  # Save with a tight layout and high resolution
        print(f"Filtering summary table saved as an image at {output_path}")


    def run(self, dataset: Dataset) -> None:
        intermediate_dir = self._create_intermediate_dir()

        original_length = self._get_total_length(dataset)
        num_samples = [original_length]
        filter_names = ["Input\nDataset"]
        remaining_samples = [original_length]
        questions_remaining_list = [self._get_total_questions(dataset)]
        questions_removed_list = [0]
        removed_samples = [0]
        filter_times_per_sample = [0]
        remaining_dataset = dataset
        
        for i, filter_i in enumerate(self.filters):
            if remaining_dataset is not None:
                num_samples_before = self._get_total_length(remaining_dataset)
                questions_before = self._get_total_questions(remaining_dataset)
            else:
                num_samples_before = original_length
                questions_before = self._get_total_questions(dataset)

            start_time = time.time()
            remaining_dataset, removed_dataset = filter_i.apply(remaining_dataset)
            check_questions_answers_length_match(remaining_dataset)
            end_time = time.time()

            num_samples_after = self._get_total_length(remaining_dataset)
            questions_after = self._get_total_questions(remaining_dataset)
            filter_time = end_time - start_time
            filter_time_per_1000_samples = 1000 * (filter_time / num_samples_before) if num_samples_before > 0 else 0

            num_samples.append(num_samples_after)
            filter_names.append(str(filter_i))
            
            remaining_samples.append(num_samples_after)
            questions_remaining_list.append(questions_after)
            removed_samples.append(num_samples_before - num_samples_after)
            questions_removed_list.append(questions_before - questions_after)
            filter_times_per_sample.append(f"{filter_time_per_1000_samples:.3f}")

            histogram_path = os.path.join(intermediate_dir, f"{str(filter_i)}_{i}_histogram.png")
            filter_i.plot_histogram(histogram_path)

            print(f"Applied filter {filter_i}. Remaining samples: {num_samples_after} ({num_samples_after / original_length * 100:.2f}%)")
            print(f"Filter time per 1000 samples: {filter_time_per_1000_samples:.3f} s")

            if self.save_intermediate == "all":
                remaining_path = os.path.join(intermediate_dir, f"{str(filter_i)}_{i}_remaining")
                removed_path = os.path.join(intermediate_dir, f"{str(filter_i)}_{i}_removed")
                remaining_dataset.save_to_disk(remaining_path)
                removed_dataset.save_to_disk(removed_path)
            elif self.save_intermediate == "subsample":
                subsampled_remaining_dataset = self._subsample_dataset(remaining_dataset, self.subsample_size)
                subsampled_removed_dataset = self._subsample_dataset(removed_dataset, self.subsample_size)
                remaining_path = os.path.join(intermediate_dir, f"{str(filter_i)}_{i}_remaining_subsample")
                removed_path = os.path.join(intermediate_dir, f"{str(filter_i)}_{i}_removed_subsample")
                
                # Save subsampled datasets
                subsampled_remaining_dataset.save_to_disk(remaining_path)
                subsampled_removed_dataset.save_to_disk(removed_path)
                
                # Visualize subsampled datasets
                remaining_viz_path = os.path.join(intermediate_dir, f"{str(filter_i)}_{i}_remaining_subsample_viz.png")
                removed_viz_path = os.path.join(intermediate_dir, f"{str(filter_i)}_{i}_removed_subsample_viz.png")
                
                asyncio.run(self._visualize_multimodal_dataset(subsampled_remaining_dataset, remaining_viz_path, max_samples=self.subsample_size))
                asyncio.run(self._visualize_multimodal_dataset(subsampled_removed_dataset, removed_viz_path, max_samples=self.subsample_size))

        self._plot_remaining_samples(num_samples, original_length, filter_names)
        self._create_summary_table(filter_names, remaining_samples, removed_samples, filter_times_per_sample, questions_remaining_list, questions_removed_list)

        final_path = os.path.join(self.output_dir, 'final_dataset')
        remaining_dataset.save_to_disk(final_path)
def main():
    parser = argparse.ArgumentParser(description="Run multimodal dataset filtering pipeline.")

    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the input dataset directory (HuggingFace format).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory where filtered dataset and outputs will be saved.")
    parser.add_argument("--sn_api_key", type=str, required=True,
                        help="SambaNova Cloud API key.")
    parser.add_argument("--sn_model", type=str, default="Meta-Llama-3.1-8B-Instruct",
                        help="SambaNova cloud model name (default: Meta-Llama-3.1-8B-Instruct)")
    parser.add_argument("--save_intermediate", choices=["none", "subsample", "all"], default="subsample",
                        help="Whether to save intermediate datasets (default: subsample)")
    parser.add_argument("--subsample_size", type=int, default=10,
                        help="Number of examples to subsample if save_intermediate=subsample (default: 10)")

    args = parser.parse_args()

    dataset = load_from_disk(args.dataset_path)

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    filters = [
        TextLengthFilterOCR(min_length=60),
        NGramFilter(n_gram=4, threshold=0.12),
        LanguageFilter(language="hu"),
        DedupQuestionsFilter(model=args.sn_model, api_key=args.sn_api_key)
    ]

    pipeline = DatasetFilteringPipeline(
        filters=filters,
        output_dir=args.output_dir,
        save_intermediate=args.save_intermediate,
        subsample_size=args.subsample_size
    )
    pipeline.run(dataset)

if __name__ == "__main__":
    main()
"""
This script is to filter existing downloaded PDFs that are not in hungarian 
"""

import argparse
import fitz  # PyMuPDF
import os 
import sys
import shutil
from tqdm import tqdm

# TODO: move paths
sys.path.append("/import/ml-sc-scratch3/nidhih/mm_hungarian/sambafeel/data/multimodal/pdf_data_extraction")
from extract_hungarian import run_fasttext_langdetect_model, FT_MODEL
from utils import pdf_page_to_text

TARGET_LANG_CODE = "hu"

def remove_pdf(pdf_path):
    """
    Return true to remove pdf if any non hungarian PDFs
    """
    doc = fitz.open(pdf_path)
    print(" --- Num of pages --- ", len(doc))
    lang = None
    for page_num in range(len(doc)):
        text = pdf_page_to_text(pdf_path, page_num)
        if text == "":
            continue
        lang, prob = run_fasttext_langdetect_model(model=FT_MODEL, text=text)
        if lang.lower() != TARGET_LANG_CODE:
            print(f"Dropping {pdf_path} due to having language {lang}")
            return True
    
    # If no text was detected, lang will stay none. Non hungarian docs were sneaking in.
    if lang is None:
        return True
    return False

def move_files_to_remove(pdf_file_to_move, warc):
    BACKUP_FOLDER = "/import/ml-sc-scratch3/nidhih/mm_hungarian/sambafeel/data/multimodal/pdf_data_scraping/outputs/single_warc_sntask/removed_from_snapshots"
    dest_folder = os.path.join(BACKUP_FOLDER, warc)
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    shutil.move(
        pdf_file_to_move,
        os.path.join(dest_folder, os.path.basename(pdf_file_to_move))
    )

def main(warc_folder):
    pdf_folder = os.path.join(warc_folder, os.path.basename((warc_folder)))
    for pdf_file in tqdm(os.listdir(pdf_folder)):
        if not pdf_file.endswith(".pdf"):
            print("Non PDF file? Why?")
            continue
        print(" --- PDF file --- ", os.path.join(pdf_folder, pdf_file))
        try:
            if remove_pdf(os.path.join(pdf_folder, pdf_file)):
                print("Removing!")
                move_files_to_remove(
                    pdf_file_to_move=os.path.join(pdf_folder, pdf_file),
                    warc=os.path.basename(warc_folder)
                    )
            else:
                print("Keeping!")
        except Exception as e:
            print(f"Exception {e} encountered, removing")
            move_files_to_remove(
                pdf_file_to_move=os.path.join(pdf_folder, pdf_file),
                warc=os.path.basename(warc_folder)
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filters PDFs."
    )
    parser.add_argument("--warc_folder", help="Path to the warc folder containing warc PDFs, jsonl file and zipped file")
    args = parser.parse_args()

    main(args.warc_folder)
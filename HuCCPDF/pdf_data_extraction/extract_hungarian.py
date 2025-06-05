import os
from PIL import Image
from datasets import Dataset, DatasetDict, load_dataset
import fitz  # PyMuPDF
import huggingface_hub
from huggingface_hub import HfApi
from utils import pdf_page_to_image, pdf_page_to_text, pdf_page_to_markdown, pdf_page_to_html
from tqdm import tqdm
import pandas as pd
import pickle
# import imagehash
import io
import hashlib
import fasttext

FASTTEXT_MODEL_PATH = "/import/ml-sc-scratch3/nidhih/mm_hungarian/downloads/lid.176.bin"
print("Loading fasttext model...")
FT_MODEL = fasttext.load_model(FASTTEXT_MODEL_PATH)
print("Done loading fasttext model...")
TARGET_LANG_CODE = "hu"

def run_fasttext_langdetect_model(model, text) -> tuple[str, float]:
    result = model.predict(text.replace("\n", ""), k=1)

    lang_code = result[0][0].split("__")[-1]
    prob = round(result[1][0], 2)
    return lang_code, prob

global all_images 
all_images = dict()

global dedup_counter 
dedup_counter = 0

def sha256(image):
    byte_stream = io.BytesIO()
    image.save(byte_stream, format=image.format)

    # Get the byte data
    image_bytes = byte_stream.getvalue()

    # Compute SHA-256 hash
    sha256_hash = hashlib.sha256(image_bytes).hexdigest()
    return sha256_hash

# print("Loading datasets...")
# decontamination_list = load_dataset('EtashGuha/HungarianDocQA', split='test') # only test, 54 manual samples
# decontamination_list2 = load_dataset('EtashGuha/RawHungarianPDFData', split='test') # test, 15.8k samples, only PDFs
# decontamination_list3 = load_dataset('nhiremath/RawHungarianPDFExtended_V2', split='test') # test, more PDFs
# print("Loaded datasets...")

# for img in decontamination_list['image']:
#     sha = sha256(img)
#     all_images[sha] = img
# print("Done loading test set 1...")

# for img in tqdm(decontamination_list2['image']):
#     sha = sha256(img)
#     all_images[sha] = img

# for img in tqdm(decontamination_list3['image']):
#     sha = sha256(img)
#     all_images[sha] = img

# print("Loaded decontamination list of images of length: ", len(all_images))  

# Function to recursively collect all PDFs in a directory
def collect_pdfs(directory):
    pdf_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
                if len(pdf_files)%500==0:
                    print(f"Collected {len(pdf_files)} PDFs")
    return pdf_files

# Function to extract text and image from each page of a PDF using the provided helper functions
def extract_text_and_images(pdf_path):
    doc = fitz.open(pdf_path)
    page_data = []
    global all_images
    for page_num in range(len(doc)):
        img = pdf_page_to_image(pdf_path, page_num, page_num)  # Using page number as index for simplicity
        sha = sha256(img)
        if sha in all_images:
            global dedup_counter
            print(dedup_counter)
            dedup_counter += 1
            continue   
        else:
            all_images[sha] = img
        text = pdf_page_to_text(pdf_path, page_num)
        if text == "":
            continue
        lang, prob = run_fasttext_langdetect_model(model=FT_MODEL, text=text)
        if lang.lower() != TARGET_LANG_CODE:
            print("dropping!!", lang)
            continue
        markdown = pdf_page_to_markdown(pdf_path, page_num)
        html = pdf_page_to_html(pdf_path, page_num)
        
        page_data.append({"text": text, "image": img, "markdown": markdown, 'html': html})

    return page_data

# Main function to process PDFs and create dataset
def create_dataset(directories, dataset_name, hf_token):
    
    all_page_data = []
    for directory in directories:
        pdf_files = collect_pdfs(directory)
        
        for pdf in tqdm(pdf_files):
            try:
                page_data = extract_text_and_images(pdf)
            except Exception as e:
                print("Skipped this because of exception ", e)
            all_page_data.extend(page_data)
            # TODO: remove total limit
            # if len(all_page_data) > 40:
            #     break

    print(" Total number of samples ----- ", len(all_page_data))
    # Prepare data for Hugging Face dataset
    data = {
        "image": [entry["image"] for entry in all_page_data],
        "text": [entry["text"] for entry in all_page_data],
        "markdown": [entry["markdown"] for entry in all_page_data],
        "html": [entry["html"] for entry in all_page_data],
        "file_name": [entry["image"] for entry in all_page_data],
    }
    
    # breakpoint()
    
    # with open("temp_again.pkl", "wb") as f:
    #     pickle.dump(data, f)

    # Save images to disk and replace with file paths
    os.makedirs(f"{dataset_name}/test", exist_ok=True)
    
    
    for i, img in tqdm(enumerate(data["file_name"])):
        img_path = f"{dataset_name}/test/page_{i}.png"
        img.save(img_path)
        data["file_name"][i] = f"page_{i}.png"

    metadata_df = pd.DataFrame(data)
    metadata_df.to_csv(f'{dataset_name}/test/metadata.csv', index=False, escapechar='\\', encoding='utf-8', errors='replace')

    # dataset = load_dataset("imagefolder", data_dir=f"{dataset_name}", split="test")
    new_dataset = {}
    new_dataset['test'] = Dataset.from_dict(data)
    new_dataset = DatasetDict(new_dataset)
    new_dataset.push_to_hub(dataset_name)




# Example usage
# directories = [
#                 "/import/ml-sc-scratch5/etashg/interleaved-datasets/hungarian_pdfs_new",
#                 "/import/ml-sc-scratch5/etashg/interleaved-datasets/hungarian_pdfs",
#                 "/import/ml-sc-scratch5/etashg/interleaved-datasets/hungarian_pdfs_long",
#                ]
# directories = [
#     # "/import/ml-sc-scratch5/etashg/interleaved-datasets/out_jonathan_hungarian",
#     # "/import/ml-sc-scratch5/etashg/interleaved-datasets/out_jonathan_hungarian_pdfs",
#     "/import/ml-sc-scratch1/etashg/interleave/out_many_jobs_hungarian/generated_hungarian_pdfs",
# ]

directories = [
    "/import/ml-sc-scratch3/nidhih/mm_hungarian/sambafeel/multimodal/pdf_data_scraping/outputs/snapshots/out_jonathan_hungarian_ss0_CC-MAIN-2020-05",
    "/import/ml-sc-scratch3/nidhih/mm_hungarian/sambafeel/multimodal/pdf_data_scraping/outputs/snapshots/out_jonathan_hungarian_ss0_CC-MAIN-2020-10", 
    "/import/ml-sc-scratch3/nidhih/mm_hungarian/sambafeel/multimodal/pdf_data_scraping/outputs/snapshots/out_jonathan_hungarian_ss0_CC-MAIN-2020-16",
    "/import/ml-sc-scratch3/nidhih/mm_hungarian/sambafeel/multimodal/pdf_data_scraping/outputs/snapshots/out_jonathan_hungarian_ss0_CC-MAIN-2020-24",
]

# dataset_name = "EtashGuha/RawHungarianPDFDataAgain"
# dataset_name = "nhiremath/RawHungarianPDFExtended_V2"
# dataset_name = "nhiremath/RawHungarianPDFExtended_V3"


hf_token = os.getenv("HF_TOKEN")
# create_dataset(directories, dataset_name, hf_token)

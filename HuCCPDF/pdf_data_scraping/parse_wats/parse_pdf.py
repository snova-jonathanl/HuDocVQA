import hashlib
import json
import re
import fitz
import os
import argparse
from tqdm import tqdm
from pebble import ProcessPool
from multiprocessing import Process, Manager
from tqdm.contrib.concurrent import process_map
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from baselines.mappers.enrichers.language_id_enrichers import load_fasttext_model, detect_lang_whole_page_fasttext
from parse_utils import distance_to_figure_coords
from multi_column import column_boxes
import io
from PIL import Image
import concurrent

### filter huge PDFs into another directory 

global lang_detect_model
lang_detect_model = load_fasttext_model()

import sys
#TODO: move paths
sys.path.append("/import/ml-sc-scratch3/nidhih/mm_hungarian/sambafeel/data/multimodal/pdf_data_extraction")
from extract_hungarian import run_fasttext_langdetect_model, FT_MODEL
from utils import pdf_page_to_text

# TODO: update these also  
# def is_lang_english(lang_detect_model, doc):
#     text = ' '.join([page.get_text("text") for page in doc])
#     return detect_lang_whole_page_fasttext(lang_detect_model, text).get('en', None) is not None

# def is_lang_japanese(lang_detect_model, doc):
#     text = ' '.join([page.get_text("text") for page in doc])
#     return detect_lang_whole_page_fasttext(lang_detect_model, text).get('ja', None) is not None


# def is_lang_hungarian(lang_detect_model, doc):
#     text = ' '.join([page.get_text("text") for page in doc])
#     return detect_lang_whole_page_fasttext(lang_detect_model, text).get('hu', None) is not None

def is_lang_hungarian(lang_detect_model, doc):
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

def matprop(m):
    """Check basic properties of an image transformation matrix.

    Supported matrices *must* have either
    - m.a == m.d == 0
    or:
    - m.b == m.c == 0
    """
    m = fitz.Matrix(m)  # ensure this is a matrix
    msg = ""
    error = False
    if m.b == m.c == 0:  # means 0/180 rotations or flippings
        if m.a * m.d > 0:  # same sign -> no flippings
            if m.a < 0:  # so both, a and d are negative!
                msg = (4, "rot 180")
            else:
                msg = (0, "nothing")
        else:  # we have a flip
            if m.a < 0:  # horizontal flip
                msg = (1, "left-right")
            else:
                msg = (2, "up-down")
    elif m.a == m.d == 0:  # means 90/270 rotations
        if m.b * m.c < 0:
            if m.b > 0:
                msg = (3, "rot 90")
            else:
                msg = (5, "rot 270")
        else:
            if m.b > 0:
                msg = (6, "up-down, rot 90")
            else:
                msg = (7, "rot 90, up-down")
    else:
        return (0, "unsupported")

    if error:
        raise ValueError("unsupported matrix")
    return msg

def recoverpix(doc, item):
    xref = item[0]  # xref of PDF image
    smask = item[1]  # xref of its /SMask

    # special case: /SMask or /Mask exists
    if smask > 0:
        pix0 = fitz.Pixmap(doc.extract_image(xref)["image"])
        if pix0.alpha:  # catch irregular situation
            pix0 = fitz.Pixmap(pix0, 0)  # remove alpha channel
        mask = fitz.Pixmap(doc.extract_image(smask)["image"])

        try:
            pix = fitz.Pixmap(pix0, mask)
        except:  # fallback to original base image in case of problems
            pix = fitz.Pixmap(doc.extract_image(xref)["image"])

        pix = fitz.Pixmap(fitz.csRGB, pix)

        if pix.color_topusage()[0] > 0.95:
            raise Exception("Image is almost monochrome")
        
        if pix.alpha:
            ext = "png"
        else:
            ext = "jpg"

        return {  # create dictionary expected by caller
            "ext": ext,
            "colorspace": pix.colorspace.n,
            "image": pix.tobytes(ext),
        }

    # special case: /ColorSpace definition exists
    # to be sure, we convert these cases to RGB PNG images
    if "/ColorSpace" in doc.xref_object(xref, compressed=True):
        pix = fitz.Pixmap(doc, xref)
        pix = fitz.Pixmap(fitz.csRGB, pix)
        
        if pix.alpha:
            ext = "png"
        else:
            ext = "jpg"
        
        return {  # create dictionary expected by caller
            "ext": ext,
            "colorspace": pix.colorspace.n,
            "image": pix.tobytes(ext),
        }
    return doc.extract_image(xref)

def process_pdf_page_fitz(doc, extracted_xref, pdf_data, page, pdf_image_dir, page_idx):
    page.clean_contents()
    bboxes = column_boxes(page, no_image_text=True)

    # bboxes is a list of fitz.IRect objects, that are sort ascending by their y0,
    # then x0 coordinates. Their text content can be extracted by all PyMuPDF
    # get_text() variants, like for instance the following:
    text_blocks = []

    for rect in bboxes:
        text_blocks.extend(page.get_text("blocks", clip=rect, sort=True, flags=fitz.TEXTFLAGS_BLOCKS & ~fitz.TEXT_PRESERVE_IMAGES))
        
    # if there are no text blocks, skip the page
    if not text_blocks:
        return pdf_data, extracted_xref
    
    valid_images = []
    
    for img_info in doc.get_page_images(page_idx):
        xref = img_info[0]
        
        if xref in extracted_xref:
            continue
        extracted_xref.add(xref)
        
        width = img_info[2]
        height = img_info[3]
        
        if min(width, height) <= 150:
            continue
        if max(width, height) >= 20000:
            continue
        
        try:
            image = recoverpix(doc, img_info)
        except Exception as e:
            print(f"Skipping {pdf_data['pdf_name']} page {page_idx} figure {xref} because it could not be recovered with exception {e}")
            continue
        
        img_bbox, img_transform = page.get_image_rects(xref, transform=True)[0]
        
        n = image["colorspace"]
        imgdata = image["image"]

        if len(imgdata) <= 2048:
            continue
        if len(imgdata) / (width * height * n) <= 0.05:
            continue
        
        img_name = f"{pdf_data['pdf_name'].split('.')[0]}_page_{page_idx}_xref_{xref}.{image['ext']}"
        imgfile = os.path.join(pdf_image_dir, "images", img_name)
        
        img_hash = hashlib.sha256()
        img_hash.update(imgdata)
        metadata = {
            "width": width,
            "height": height,
            "sha256": img_hash.hexdigest()
        }
        
        pil_img = Image.open(io.BytesIO(imgdata))
        if matprop(img_transform)[0] == 2:
            pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
        elif matprop(img_transform)[0] == 1:
            pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
        elif matprop(img_transform)[0] == 3:
            pil_img = pil_img.transpose(Image.ROTATE_90)
        elif matprop(img_transform)[0] == 5:
            pil_img = pil_img.transpose(Image.ROTATE_270)
        elif matprop(img_transform)[0] == 6:
            pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.ROTATE_90)
        elif matprop(img_transform)[0] == 7:
            pil_img = pil_img.transpose(Image.ROTATE_90).transpose(Image.FLIP_TOP_BOTTOM)
        
        # with open(imgfile, "wb") as f:
        #     f.write(imgdata)
        pil_img.save(imgfile)
        
        # write metadata to a file
        with open(imgfile.replace(f".{image['ext']}", ".json"), "w") as f:
            f.write(json.dumps(metadata))
        
        valid_images.append((img_name, img_bbox))

    # text block to closest image mapping
    text_to_image = defaultdict(list)
    for image_filename, image in valid_images:
        closest_text_block = None
        closest_distance = float("inf")
        for text_block in text_blocks:
            distance = distance_to_figure_coords(text_block[0], text_block[1], text_block[2], text_block[3], image[0], image[1], image[2], image[3])
            if distance < closest_distance:
                closest_text_block = text_block
                closest_distance = distance
        if closest_text_block:
            text_to_image[(closest_text_block[0], closest_text_block[1], closest_text_block[2], closest_text_block[3])].append(image_filename)

    for text_block in text_blocks:
        string_block = text_block[4].strip("\n").replace("-\n", "")
        string_block = re.sub(r'(\w)\n(\w)', r'\1 \2', string_block)

        text_block_coords = (text_block[0], text_block[1], text_block[2], text_block[3])

        for image_filename in text_to_image.get(text_block_coords, []):
            pdf_data["images"].append(image_filename)
            pdf_data["texts"].append(None)

        if pdf_data["texts"] and isinstance(pdf_data["texts"][-1], str):
            pdf_data["texts"][-1] += string_block if pdf_data["texts"][-1].endswith(" ") else " " + string_block
        else:
            pdf_data["texts"].append(string_block)
            pdf_data["images"].append(None)

    return pdf_data, extracted_xref

def process_pdf(inputa):
    output_dir, pdf_path, pdf_url, max_num_pages = inputa
    # Initialize the output data structure
    pdf_data = {
        "pdf_name": os.path.basename(pdf_path),
        "url": pdf_url,
        "texts": [],
        "images": []
    }
    
    # Open the PDF file
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        return None

    if doc.page_count > max_num_pages:
        return None

    # if not is_lang_hungarian(lang_detect_model, doc):
    #     print("Removing doc")
    #     os.remove(pdf_path)
    #     return None    
        
    extracted_xref = set()
    
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        
    for page_idx, page in enumerate(doc):
        pdf_data, extracted_xref = process_pdf_page_fitz(doc, extracted_xref, pdf_data, page, output_dir, page_idx)
 
    for i in range(len(pdf_data["texts"])):
        if pdf_data["texts"][i]:
            pdf_data["texts"][i] = re.sub(r'\s+', ' ', re.sub(r'(?<=[\.\?\!])\n', ' ',pdf_data["texts"][i]))
    
    if not any(pdf_data["images"]):
        return None

    return pdf_data

def process_pdf_with_timeout(pdf_args, output_queue):
    try:
        pdf_result = process_pdf(*pdf_args)
        output_queue.put(pdf_result)
    except Exception as e:
        print(f"Skipping {pdf_args[1]} because it could not be processed with exception {e}")
        output_queue.put(None)

def worker(input_queue, output_dir, max_num_pages, max_seconds_per_pdf, progress_update, results_queue):
    while True:
        pdf_path, pdf_url = input_queue.get()
        print(pdf_path)
        if pdf_path is None:  # Sentinel value to stop the worker
            break
        output_queue = Manager().Queue()
        process_pdf_with_timeout((output_dir, pdf_path, pdf_url, max_num_pages), output_queue)
        
        pdf_process = Process(target=process_pdf_with_timeout, args=((output_dir, pdf_path, pdf_url, max_num_pages), output_queue))
        pdf_process.start()
        pdf_process.join(timeout=max_seconds_per_pdf)
        print("finished")
        if pdf_process.is_alive():
            pdf_process.terminate()
            print(f"Terminated processing of {pdf_path} due to timeout.")
        else:
            pdf_result = output_queue.get()
            if pdf_result:
                results_queue.put(pdf_result)
        progress_update.put(1)
                
def extract_text_and_images(input_dir, pdf_urls, output_dir, num_workers, max_num_pages, max_seconds_per_pdf):
    
    os.makedirs(output_dir, exist_ok=True)
    pdf_paths = [os.path.join(input_dir, pdf) for pdf in os.listdir(input_dir) if pdf.endswith('.pdf')]
    
    pdf_idx_to_url = {}
    if pdf_urls:
        with open(pdf_urls, 'r') as file:
            for pdf_idx, line in enumerate(file):
                data = json.loads(line)
                pdf_idx_to_url[pdf_idx] = data['url']
    
    with Manager() as manager:
        input_queue = manager.Queue()
        progress_update = manager.Queue()
        results_queue = manager.Queue()
        processes = []

        all_inputs = []
        # Enqueue tasks
        for pdf_path in pdf_paths:
            pdf_url = pdf_idx_to_url[int(os.path.basename(pdf_path).replace(".pdf", ""))] if pdf_urls else None
            all_inputs.append((output_dir, pdf_path, pdf_url, max_num_pages))
        with tqdm(total=len(all_inputs)) as pbar:
            def update_progress_bar(x):
                print("update")
                pbar.update()
                return x
            with ProcessPool(max_workers=num_workers) as pool:
                future = pool.map(process_pdf, all_inputs, timeout=max_seconds_per_pdf, callback=update_progress_bar)
        # Collect results
        results = []
        all_results = future.result()
        while True:
            try:
                results.append(all_results.next())
            except concurrent.futures._base.TimeoutError as ex:
                pass
            except StopIteration as ex:
                break
            except Exception as e:
                print("An error occurred:", e)
                pass
        pool.join()
        pool.close()
        
        with open(os.path.join(output_dir, f"{os.path.basename(input_dir)}.jsonl"), "w") as f:
            for result in results:
                if result is not None:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process PDF files and extract text and images in parallel."
    )
    parser.add_argument("--input_dir", required=True, help="Path to the input directory containing PDF files")
    parser.add_argument("--pdf_urls", help="Path to the JSONL file containing PDF URLs", default=None)
    parser.add_argument("--output_dir", required=True, help="Path to the output directory")
    parser.add_argument("--num_workers", type=int, required=True, help="Number of parallel processes")
    parser.add_argument("--max_num_pages", type=int, required=True, help="Maximum number of pages to process")
    parser.add_argument("--max_seconds_per_pdf", type=int, required=True, help="Maximum number of seconds to process a PDF")
    args = parser.parse_args()

    extract_text_and_images(args.input_dir, args.pdf_urls, args.output_dir, args.num_workers, args.max_num_pages, args.max_seconds_per_pdf)
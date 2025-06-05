import argparse
import requests
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
import os
import json
from tqdm import tqdm
import magic

##
import fitz
from parse_pdf import is_lang_hungarian
from baselines.mappers.enrichers.language_id_enrichers import load_fasttext_model, detect_lang_whole_page_fasttext


"""
Drop all HTML files - doc how often this happens + delete
"""

global lang_detect_model
lang_detect_model = load_fasttext_model()

def filter_pdf(pdf_path, max_num_pages=50):
    # True is pdf_path was filtered and discarded
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        ## remove these???
        print(f"Failed to read file {pdf_path} , removing.")
        os.remove(pdf_path)
        return True

    if doc.page_count > max_num_pages:
        # ignore large PDF files for now
        print(f"Dropping {pdf_path} because it has more than {max_num_pages} pages.")
        return False

    print("File type -- ", magic.from_file(pdf_path), " from --- ", pdf_path)
    if "HTML" in magic.from_file(pdf_path) or "html" in magic.from_file(pdf_path):
        print("Dropping because HTML file")
        os.remove(pdf_path)
        return True

    if not is_lang_hungarian(lang_detect_model, doc):
        print("Removing doc")
        os.remove(pdf_path)
        return True

    return False   
    

def download_pdf(params):
    url, output_directory, file_index = params
    
    try:
        with requests.get(url, stream=True, timeout=10) as response:
            if response.status_code == 200:
                content_length = response.headers.get('Content-Length')
                if content_length is not None:
                    size = int(content_length)
                                        
                    # ignore the file if the size is greater than 50MB (probably just a bunch of images)
                    if size > 50000000:
                        return

                    with open(os.path.join(output_directory, f"{file_index:08}.pdf"), 'wb') as f:
                        f.write(response.content)
                    return os.path.join(output_directory, f"{file_index:08}.pdf")
                    
    except:
        return

def worker(params):
    total_number_of_downloaded_pdfs, removed_pdfs = 0, 0
    print("downloading")
    urls, output_directory, num_threads, start_index = params
    
    futures = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i, url in enumerate(urls):
            worker_params = (url, output_directory, i + start_index)
            future = executor.submit(download_pdf, worker_params)
            futures.append(future)

        with tqdm(total=len(urls), desc=f"Downloading from thread {start_index}") as progress:
            for future in futures:
                try:
                    pdf_path = future.result(timeout=15)
                    total_number_of_downloaded_pdfs += 1
                    if pdf_path is not None:
                        # check for language. if not, remove
                        if filter_pdf(pdf_path):
                            removed_pdfs += 1

                except Exception as exc:
                    ## this could have irrelevant PDFs, keeping so they can be removed later
                    print(f'A PDF downloading task generated an exception: {exc}')
                finally:
                    progress.update(1)
            print(" Total number of downloaded PDFs: ", total_number_of_downloaded_pdfs)
            print(" Total number of removed PDFs: ", removed_pdfs)

def main(jsonl_file, num_processes, num_threads, output_directory):
 
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Read URLs from the JSONL file
    pdf_urls = []
    with open(jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            pdf_urls.append(data['url'])
                                                              
    chunk_size = len(pdf_urls) // num_processes
    chunks = [pdf_urls[i:i + chunk_size] for i in range(0, len(pdf_urls), chunk_size)]

    params = [(chunk, output_directory, num_threads, i*chunk_size) for i, chunk in enumerate(chunks)]
    # worker(params[0])
    with Pool(processes=num_processes) as pool:
        pool.map(worker, params)
    pool.join()
    pool.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download PDFs in parallel."
    )
    parser.add_argument("--jsonl_file", help="Path to the JSONL file containing URLs")
    parser.add_argument("--num_processes", type=int, default=8, help="Number of processes")
    parser.add_argument("--num_threads", type=int, default=10, help="Number of threads per process")
    parser.add_argument("--output_dir", help="Path to the output directory")
    args = parser.parse_args()

    main(args.jsonl_file, args.num_processes, args.num_threads, args.output_dir)
import argparse
from parse_single_wat import process_wat_file
import os
import subprocess
from download_pdfs import main as download_pdf
from parse_pdf import extract_text_and_images
from cleanup import main as clean
from baselines.core.processor import process_single_file
from yaml import safe_load
# import wandb
import time

def get_file_names(file_url, local_storage_path):
    wat_file = os.path.basename(file_url) # example: CC-MAIN-20230921210007-20230922000007-00007.warc.wat.gz
    jsonl_file_basename = os.path.splitext(os.path.splitext(wat_file)[0])[0]

    # download the warc.wat.gz file locally
    if not os.path.exists(os.path.join(local_storage_path, wat_file)) or os.path.getsize(os.path.join(local_storage_path, wat_file))<100:
        # subprocess.run(["curl", "--retry", "1000", "--retry-delay", "1", "-o", os.path.join(local_storage_path, wat_file), file_url], check=True)
        # curl was failing with an SSL error
        subprocess.run(["wget", file_url, "-O", os.path.join(local_storage_path, wat_file)])
    else:
        print("File already exists")

    if os.path.exists(os.path.join(local_storage_path, wat_file)):
        jsonl_file = f"{jsonl_file_basename}_urls.jsonl"
        return jsonl_file
    else:
        return None

def check_complete_log_file(wat_log_location):
    # print(" Checking wat log location ---- ", wat_log_location)
    if os.path.exists(wat_log_location):
        with open(wat_log_location, 'r') as f:
            logfile = f.read()
            download100 = logfile.split("Downloading from thread 0: 100%")
            # If this line has occurred at least once, we have finished downloading all post processing happens immediately after
            if len(download100)>=2:
                return True
    if os.path.exists(wat_log_location.replace(f".log", "_run.log")):
        with open(wat_log_location.replace(f".log", "_run.log"), 'r') as f:
            logfile = f.read()
            download100 = logfile.split("Downloading from thread 0: 100%")
            # If this line has occurred at least once, we have finished downloading all post processing happens immediately after
            if len(download100)>=2:
                return True

    return False


def main(num_workers, num_download_threads, storage_path, wat_file, log_file):
    if check_complete_log_file(log_file):
        print(f"Done processing at {storage_path}")
        return 
    
    print("Not processed yet! Continuing with ", wat_file)
    print("Storage path to be saved to ---- ", storage_path)

    print(f"NUM {num_workers}, Dow {num_download_threads}, sto: {storage_path},wat {wat_file}")
    t1 = time.time()
    if not os.path.exists(storage_path):
        os.makedirs(storage_path, exist_ok=True)
        
    jsonl_file = get_file_names(wat_file, storage_path)
    t2 = time.time()
    if jsonl_file is None:
        print("WARNING: wat log file jsonl file is None")
        return
    
    wat_file = os.path.basename(wat_file)
    jsonl_file_basename = os.path.splitext(os.path.splitext(wat_file)[0])[0]
    print("Jsonl file name: ", jsonl_file_basename)
    
    t3 = time.time()
    # Get list of URLs of PDFs, save in jsonl file
    process_wat_file(os.path.join(storage_path, wat_file), "document", os.path.join(storage_path, jsonl_file))
    
    t4 = time.time()
    download_pdf( os.path.join(storage_path, jsonl_file), 1, num_download_threads, os.path.join(storage_path, jsonl_file_basename))
    # breakpoint()
    t5 = time.time()
    # extract_text_and_images(os.path.join(storage_path, jsonl_file_basename), os.path.join(storage_path, jsonl_file), f"{os.path.join(storage_path, jsonl_file_basename)}_pdfs_processed", 32, 50, 30)
    # t6 = time.time()
    # ### yaml = "dclm/baselines/baselines_configs/refinedweb.yaml"
    # yaml = "/import/ml-sc-scratch3/nidhih/mm_hungarian/sambafeel/multimodal/pdf_data_scraping/parse_wats/baselines/baselines_configs/refinedweb.yaml"
    
    # with open(yaml, "r") as yaml_file:
    #     config_data = safe_load(yaml_file)
    #     config_data = {v["source"]: v for v in config_data}
    # os.chdir("dcnlp")
    # t7 = time.time()
    # process_single_file(config_data=config_data,
    #                     raw_data_dirpath=f"{os.path.join(storage_path, jsonl_file_basename)}_pdfs_processed",
    #                     jsonl_relpath=f"{jsonl_file_basename}.jsonl",
    #                     source_name="cc",
    #                     base_output_path=f"{os.path.join(storage_path, jsonl_file_basename)}_pdfs_processed",
    #                     workers=1,
    #                     overwrite=False)
    
    # os.chdir("../")
    # raw_jsonl_path = f"{os.path.join(storage_path, jsonl_file_basename)}_pdfs_processed"
    # raw_jsonl_path = os.path.join(raw_jsonl_path, f'processed_data/{jsonl_file_basename}_processed.jsonl')
    # t8 = time.time()
    # clean(raw_jsonl_path, os.path.join(f"{os.path.join(storage_path, jsonl_file_basename)}_pdf_processed/images"))
    # t9 = time.time()
    
    result = {
        'get_file_names': t2 - t1,
        'process_wat_file': t4 - t3,
        "download_time": t5 - t4,
        # "extract_text_and_images": t6 - t5,
        # "processing_single_file": t8 - t7,
        # 'cleanup': t9 - t8
    }
    
    # wandb.log(result)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process WAT files and extract web documents."
    )
    parser.add_argument("--num_workers", type=int, help="num workers")
    parser.add_argument("--num_download_threads", type=int, help="Number of Download Threads")
    parser.add_argument("--storage_path", type=str)
    parser.add_argument("--wat_file", help="Path to the WAT file")
    parser.add_argument("--log_file", help="Path to the WAT files' log")
    args = parser.parse_args()
    # wandb.init(id="wla74ozz", project="interleaved-datasets")
    
    main(args.num_workers, args.num_download_threads, args.storage_path, args.wat_file, args.log_file)
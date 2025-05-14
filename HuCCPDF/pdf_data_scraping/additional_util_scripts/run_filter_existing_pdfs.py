### Usage: python run_filter_existing_pdfs.py --source_dir=/import/ml-sc-scratch3/nidhih/mm_hungarian/sambafeel/data/multimodal/pdf_data_scraping/outputs/single_warc_sntask/snapshots/out_jonathan_hungarian_ss0_CC-MAIN-2021-17/

import argparse
import os
import shutil
import time
from subprocess import call

SLEEP_TIME_BETWEEN_SUBMIT_BATCHES=4*60 # mins per filtering job
MAX_NUM_PARALLEL_TASKS_TO_LAUNCH=30

LOG_LOCATION = "/import/ml-sc-scratch3/nidhih/mm_hungarian/sambafeel/data/multimodal/pdf_data_scraping/outputs/single_warc_sntask/removed_from_snapshots/logs"

def main(source_dir):

    warc_folders = [fold for fold in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, fold))]
    print(f"Total number of wat files passed: {len(warc_folders)}")

    i = 0
    while i<len(warc_folders):
        job_submit_counter = 0

        while job_submit_counter <= MAX_NUM_PARALLEL_TASKS_TO_LAUNCH:
            warc_f = warc_folders[i]
            warc_log_location = os.path.join(LOG_LOCATION, warc_f+"_filter.log")
            print(f"Processing WARC file --- {warc_f}")
            call([
                'sntask', 'run', '-j', 'filter_hungarian_pdf', '--timeout', '1024:00:00', '--cpus-per-task', 
                '2', '--host-mem', '100G', '--inherit-env', '--outfile', warc_log_location, 
                '--', 'python', 'filter_existing_pdfs.py', '--warc_folder', os.path.join(source_dir, warc_f)]
                )
            job_submit_counter += 1
            i += 1

        print(f"Done submitting {job_submit_counter} to process, now waiting {SLEEP_TIME_BETWEEN_SUBMIT_BATCHES} seconds before processing next batch.")
        time.sleep(SLEEP_TIME_BETWEEN_SUBMIT_BATCHES)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter existing files for only hungarian docs."
    )
    parser.add_argument("--source_dir", help="Directory with sub directories of wat files")
    args = parser.parse_args()
    
    main(args.source_dir)
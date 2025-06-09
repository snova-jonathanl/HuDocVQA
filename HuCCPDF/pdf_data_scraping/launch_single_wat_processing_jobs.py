import argparse
import os
import shutil
import time
from subprocess import call


def check_complete_log_file(wat_log_location):
    # print(" Checking wat log location: ", wat_log_location)
    if os.path.exists(wat_log_location):
        with open(wat_log_location, 'r') as f:
            logfile = f.read()
            download100 = logfile.split("Downloading from thread 0: 100%")
            # If this line has occurred at least once, we have finished downloading all post processing happens immediately after
            if len(download100)>=2:
                return True
    # Log files of 2 types exist for this run
    if os.path.exists(wat_log_location.replace(f".log", "_run.log")):
        with open(wat_log_location.replace(f".log", "_run.log"), 'r') as f:
            logfile = f.read()
            download100 = logfile.split("Downloading from thread 0: 100%")
            # If this line has occurred at least once, we have finished downloading all post processing happens immediately after
            if len(download100)>=2:
                return True

    return False


NUM_THREADS = 4
NUM_WORKERS = 4
MAX_NUM_PARALLEL_TASKS_TO_LAUNCH = int(os.environ.get("MAX_NUM_PARALLEL_WARC_PROCESSING_TASKS", default=30)) # 30 
SLEEP_TIME_BETWEEN_SUBMIT_BATCHES = int(os.environ.get("SLEEP_TIME_BETWEEN_WARC_PROCESSING_BATCHES", default=3000)) # 3000 # 50 mins
DEFAULT_STORAGE_PATH = os.environ.get("PDF_STORAGE_PATH")
WARC_PROCESSING_LOG_PATH = os.environ.get("WARC_PROCESSING_LOG_PATH")

def main(wat_files):
    print(f"Total number of wat files passed: {len(wat_files)}")

    i = 0
    while i<len(wat_files):
        job_submit_counter = 0

        while job_submit_counter <= MAX_NUM_PARALLEL_TASKS_TO_LAUNCH:
            wat_file = wat_files[i]
            warc_file_name = wat_file.split("/")[-1].replace(".wat.gz", "")
            print(f"Processing WARC file --- {warc_file_name}")
            local_storage_path = os.path.join(DEFAULT_STORAGE_PATH, warc_file_name)
            ### TODO: Change this depending on which run. Will be single place once pipeline is finalized
            warc_log_location = os.path.join(WARC_PROCESSING_LOG_PATH, warc_file_name+".wat.gz.log")
            if check_complete_log_file(warc_log_location):
                print(f"Log file at {warc_log_location} complete, skipping...")
            else:
                print(f"Going to continue with {wat_file}")
                call([
                    'sntask', 'run', '-j', 'hungarian_pdf', '--timeout', '1024:00:00', '--cpus-per-task', 
                    '32', '--host-mem', '100G', '--inherit-env', '--outfile', warc_log_location.replace(f".log", f"_run.log"), 
                    '--', 'python', 'parse_wats/everything_for_single_wat.py', '--num_workers', str(NUM_WORKERS), 
                    '--num_download_threads', str(NUM_THREADS), '--storage_path', local_storage_path, 
                    '--wat_file', wat_file, '--log_file', warc_log_location]
                    )
                job_submit_counter += 1
            i += 1

        print(f"Done submitting {job_submit_counter} to process, now waiting {SLEEP_TIME_BETWEEN_SUBMIT_BATCHES} seconds before processing next batch.")
        time.sleep(SLEEP_TIME_BETWEEN_SUBMIT_BATCHES)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process WAT files and extract web documents."
    )
    parser.add_argument("--wat_files", nargs='+', help="Array to the WAT files")
    args = parser.parse_args()
    
    main(args.wat_files)
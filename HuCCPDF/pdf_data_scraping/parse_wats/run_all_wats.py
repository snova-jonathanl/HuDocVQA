import sys
import os
import subprocess
import boto3
from everything_for_single_wat import main as process_wat
import argparse
from functools import partial
from multiprocessing import Pool, freeze_support
     
if __name__ == '__main__':
    freeze_support()
    
    wandb.init(project="interleaved-datasets")
    parser = argparse.ArgumentParser(description='Process Common Crawl data.')
    parser.add_argument('--number_of_workers', type=int, help='Number of workers')
    parser.add_argument('--number_of_download_threads', type=int, help='Number of download threads')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--portion_index', type=int, help='Portion index')
    parser.add_argument('--cc_snapshot', type=str, help='Common Crawl snapshot')
    parser.add_argument('--total_shards', type=int, help='Total number of shards')
    parser.add_argument('--portions', type=int, help='Number of portions')

    args = parser.parse_args()

    NUM_WORKERS = args.number_of_workers
    NUM_DOWNLOAD_THREADS = args.number_of_download_threads
    STORAGE_PATH = args.output_dir
    PORTION_INDEX = args.portion_index
    CC_SNAPSHOT = args.cc_snapshot
    TOTAL_SHARDS = args.total_shards
    PORTIONS = args.portions

    if STORAGE_PATH.startswith("s3://"):
        LOCAL_STORAGE_PATH = os.path.basename(STORAGE_PATH)
    else:
        LOCAL_STORAGE_PATH = STORAGE_PATH

    os.makedirs(LOCAL_STORAGE_PATH, exist_ok=True)

    BASE_BUCKET = "s3://commoncrawl/crawl-data/{}/segments/".format(CC_SNAPSHOT)

    SHARDS_PER_PORTION = TOTAL_SHARDS // PORTIONS
    REMAINING_SHARDS = TOTAL_SHARDS % PORTIONS
    START_IDX = PORTION_INDEX * SHARDS_PER_PORTION

    if PORTION_INDEX == PORTIONS:
        END_IDX = START_IDX + SHARDS_PER_PORTION + REMAINING_SHARDS - 1
    else:
        END_IDX = START_IDX + SHARDS_PER_PORTION - 1

    print("Number of workers:", NUM_WORKERS)
    print("Number of download threads:", NUM_DOWNLOAD_THREADS)
    print("Storage Path:", STORAGE_PATH)
    print("Local Storage Path:", LOCAL_STORAGE_PATH)
    print("Processing files from index", START_IDX, "to", END_IDX)

    
    s3 = boto3.client('s3')
    wats_to_process = []

    current_idx = 0

    segments = s3.list_objects(Bucket="commoncrawl", Prefix="crawl-data/{}/segments/".format(CC_SNAPSHOT), Delimiter="/")['CommonPrefixes']
    for segment in segments:
        segment_key = segment['Prefix']
        wat_files = s3.list_objects_v2(Bucket="commoncrawl", Prefix=segment_key + "wat/")
        for wat_file in wat_files["Contents"]:
            wat_key = wat_file['Key']
            if current_idx >= START_IDX and current_idx <= END_IDX:
                wats_to_process.append("https://data.commoncrawl.org/" + wat_key)
            if current_idx >= END_IDX:
                break
            current_idx += 1
        if current_idx >= END_IDX:
            break

    # download_and_process(wats_to_process[5])

    # # Uncomment the following lines to run the function in parallel
    from multiprocessing import Pool
    with Pool(processes=NUM_WORKERS) as pool:
        pool.map(partial(process_wat, NUM_WORKERS, NUM_DOWNLOAD_THREADS, LOCAL_STORAGE_PATH), wats_to_process)

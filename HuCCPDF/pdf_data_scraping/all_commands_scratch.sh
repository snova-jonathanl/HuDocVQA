#!/bin/bash

## For each snapshot from the array of snapshots below, this script runs parse_wats.sh. 
## Prior requirements: AWS CLI setup to access files of a shard
## conda env: conda activate /import/snvm-sc-scratch1/nidhih/conda/envs/transformer_latest_clone1 

# Array of snapshots
# SNAPSHOTS=(
#     "CC-MAIN-2020-05"
#     "CC-MAIN-2020-10"
#     "CC-MAIN-2020-16"
# )
# SNAPSHOTS=(
#     "CC-MAIN-2020-24"
#     "CC-MAIN-2020-29"
#     "CC-MAIN-2020-34"
#     "CC-MAIN-2020-40"
#     "CC-MAIN-2020-45"
#     "CC-MAIN-2020-50"
# )
# SNAPSHOTS=(
#     "CC-MAIN-2021-04"
#     "CC-MAIN-2021-10"
#     "CC-MAIN-2021-17"
#     "CC-MAIN-2021-21"
#     "CC-MAIN-2021-25"
#     "CC-MAIN-2021-31"
#     "CC-MAIN-2021-39"
#     "CC-MAIN-2021-43"
#     "CC-MAIN-2021-49"
# )
# SNAPSHOTS=(
#     "CC-MAIN-2021-04"
# )
# SNAPSHOTS=(
#     "CC-MAIN-2021-10"
# )
# SNAPSHOTS=(
#     "CC-MAIN-2021-04"
# )

## each warc wat file is its own sntask 
# SNAPSHOTS=(
#     "CC-MAIN-2021-17"
# )

SNAPSHOTS=(
    "CC-MAIN-2021-17"
)

export PDF_STORAGE_PATH="/import/ml-sc-scratch3/nidhih/mm_hungarian/sambafeel/multimodal/pdf_data_scraping/outputs/single_warc_sntask/snapshots/out_jonathan_hungarian_ss0_CC-MAIN-2021-17"
export WARC_PROCESSING_LOG_PATH="/import/ml-sc-scratch3/nidhih/mm_hungarian/sambafeel/multimodal/pdf_data_scraping/outputs/single_warc_sntask/logs/moved_logs"
export MAX_NUM_PARALLEL_WARC_PROCESSING_TASKS=30
export SLEEP_TIME_BETWEEN_WARC_PROCESSING_BATCHES=3000 # seconds

# Loop through each snapshot and run parsing
for SNAPSHOT in "${SNAPSHOTS[@]}"
do
    echo "Processing snapshot: $SNAPSHOT"
    
    # On each snapshot, run parse_wats.sh <number_of_workers> <number_of_download_threads> <output_dir> <portion_index> <startingshard> <name of snapshot>
    sntask run -j hungarian_pdf --timeout 1024:00:00 --cpus-per-task 72 --host-mem 100G --inherit-env \
    --outfile /import/ml-sc-scratch3/nidhih/mm_hungarian/sambafeel/multimodal/pdf_data_scraping/outputs/single_warc_sntask/sntask_hungarian_ss0_rr_deleted_$SNAPSHOT.log \
    -- bash parse_wats.sh 8 8 /import/ml-sc-scratch3/nidhih/mm_hungarian/sambafeel/multimodal/pdf_data_scraping/outputs/single_warc_sntask/snapshots/out_jonathan_hungarian_ss0_$SNAPSHOT 0 0 $SNAPSHOT
    
    echo "Completed processing for snapshot: $SNAPSHOT"
    echo "----------------------------------------"
done

echo "All snapshots have been processed."
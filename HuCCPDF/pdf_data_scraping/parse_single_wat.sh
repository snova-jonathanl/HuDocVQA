#!/bin/bash

SNAPSHOT_WAT=$1
LOCAL_STORAGE_PATH=$2
NUM_WORKERS=$3
NUM_DOWNLOAD_THREADS=$4

WAT_LOG_LOCATION=/import/ml-sc-scratch3/nidhih/mm_hungarian/sambafeel/multimodal/pdf_data_scraping/outputs/single_warc_sntask/logs/sntask_hungrian_ss0_$SNAPSHOT_WAT.log

sntask run -j hungarian_pdf --timeout 1024:00:00 --cpus-per-task 32 --host-mem 100G --inherit-env \
--outfile $WAT_LOG_LOCATION \
-- python parse_wats/everything_for_single_wat.py --num_workers $NUM_WORKERS --num_download_threads $NUM_DOWNLOAD_THREADS --storage_path $LOCAL_STORAGE_PATH --wat_file $SNAPSHOT_WAT --wat_log_location $WAT_LOG_LOCATION

echo "Completed processing for snapshot: $SNAPSHOT_WAT"
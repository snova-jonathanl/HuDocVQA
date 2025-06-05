#!/bin/bash

# Usage: sh script_name <number_of_workers> <number_of_download_threads> <output_dir> <portion_index> <startingshard>
# export PATH=“$PATH:/export/home/repos/conda_envs/openflamingo”

if [ "$#" -lt 6 ] || [ "$#" -gt 6 ]; then
    echo "Usage: $0 <number_of_workers> <number_of_download_threads> <output_dir> <portion_index>"
    exit 1
fi

NUM_WORKERS=$1 # 72
NUM_DOWNLOAD_THREADS=$2 # 72
STORAGE_PATH=$3 # string - path
STARTING_SHARD_PER_HUNDRED=$5 # 0
wats_to_process=()

echo $STARTING_SHARD_PER_HUNDRED #0
STARTINGSEQ=$((STARTING_SHARD_PER_HUNDRED * 100 + 1))
ENDINGSEQ=$(((STARTING_SHARD_PER_HUNDRED + 1) * 100))
echo $STARTINGSEQ #1
echo $ENDINGSEQ #100

MAX_PARALLEL_WATS_TO_PROCESS=50

PORTION_INDEX=$portshard
if [[ $STORAGE_PATH == s3://* ]]; then
    LOCAL_STORAGE_PATH=$(basename $STORAGE_PATH)
else
    LOCAL_STORAGE_PATH=$STORAGE_PATH
fi

mkdir -p "$LOCAL_STORAGE_PATH" # name of crawl

CC_SNAPSHOT=${6:-"CC-MAIN-2023-40"} 

echo "Accessing Common Crawl snapshot: $CC_SNAPSHOT"
BASE_BUCKET="s3://commoncrawl/crawl-data/${CC_SNAPSHOT}/segments/"
TOTAL_SHARDS=90000 ## where did this come from?
PORTIONS=50

# Calculate the index range based on the portion index
SHARDS_PER_PORTION=$((TOTAL_SHARDS / PORTIONS))
REMAINING_SHARDS=$((TOTAL_SHARDS % PORTIONS))
# START_IDX=$(( STARTINGSEQ * SHARDS_PER_PORTION )) ## change to 0 or 1?
START_IDX=1
END_IDX=$(( ENDINGSEQ * SHARDS_PER_PORTION  - 1))
# END_IDX=3


echo "Number of workers: $NUM_WORKERS"
echo "Number of download threads: $NUM_DOWNLOAD_THREADS"
echo "Storage Path: $STORAGE_PATH"
echo "Local Storage Path: $LOCAL_STORAGE_PATH"
echo "Processing files from index $START_IDX to $END_IDX"


export LOCAL_STORAGE_PATH
export NUM_DOWNLOAD_THREADS
export STORAGE_PATH

# Generating the list of WARC files to process within the specified index range
# save to wats_to_process 

# current_idx=0
# echo 
# segments=$(aws s3 ls "${BASE_BUCKET}" | awk '{print $2}')
# for segment in $segments; do
#     echo "${BASE_BUCKET}${segment}wat/"
#     wat_files=$(aws s3 ls "${BASE_BUCKET}${segment}wat/" | awk '{print $4}')
#     for wat_file in $wat_files; do
#         # echo "Current index: $current_idx" 
#         if [ $current_idx -ge $START_IDX ] && [ $current_idx -le $END_IDX ]; then
#             wats_to_process+=("https://data.commoncrawl.org/crawl-data/${CC_SNAPSHOT}/segments/${segment}wat/${wat_file}")
#         fi
#         if [ $current_idx -ge $END_IDX ]; then
#             break 2
#         fi
#         ((current_idx++))
#     done
#     echo 
# done

## Run python script for each subsegment of warc
current_idx=0
echo 
segments=$(aws s3 ls "${BASE_BUCKET}" | awk '{print $2}')
for segment in $segments; do
    echo "${BASE_BUCKET}${segment}wat/"
    wat_files=$(aws s3 ls "${BASE_BUCKET}${segment}wat/" | awk '{print $4}')
    warc_files_in_segment=()
    for wat_file in $wat_files; do
        warc_files_in_segment+=("https://data.commoncrawl.org/crawl-data/${CC_SNAPSHOT}/segments/${segment}wat/${wat_file}")
        # echo "${wat_file}"
    done
    # Often about 
    echo "Starting..."
    python launch_single_wat_processing_jobs.py --wat_files "${warc_files_in_segment[@]}" 
done

# usually 100kish wat files over all segments


# Function to download, process, and upload a single WARC file
# download_and_process() {
#     python parse_wats/everything_for_single_wat.py --num_workers $NUM_WORKERS --num_download_threads $NUM_DOWNLOAD_THREADS --storage_path $LOCAL_STORAGE_PATH --wat_file $1
# }

## run these parallely, sntask for each. 
## Each warc file sntask, separate log files so no locks happen
## Reduce number of threads to 8 max
# export -f download_and_process
# python launch_single_wat_processing_jobs.py --wat_files "${wats_to_process[@]}" 
# for i in $(seq 1 ${#wats_to_process[*]});
# do
#     # echo ${#wats_to_process[*]} # total len
#     echo "Currently at wat from wat_to_process: ${wats_to_process[$i]}"
#     # download_and_process "${wats_to_process[$i]}"
#     #### parse_single_wat.sh wat local_storage_path num_workers num_download_threads
#     bash parse_single_wat.sh ${wats_to_process[$i]} $LOCAL_STORAGE_PATH 4 4
#     if $i -ge 2; then
#         break
#     fi
# done

# echo "Found ${#wats_to_process[@]} WARC files to process (from index $START_IDX to $END_IDX)"

# printf "%s\n" "${wats_to_process[@]}" | parallel -j "$NUM_WORKERS" --ungroup download_and_process {}
import os
import glob
import argparse
import jsonlines
import subprocess
from datasets import load_dataset, Dataset, DatasetDict
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
from pathlib import Path

required_fields = [
    'image',
    'text',
    'questions',
    'answers',
    'ocr'
]

def get_num_examples(filepath):
    subp_output = subprocess.check_output(f'cat {filepath} | wc -l', shell=True, text=True).strip()
    return int(subp_output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, help='Output directory specified by generate_synqa.py. Assumes a subdirectory for each dataset split, and a corresponding `jsonl` file and `images` subdirectory for each split.')
    parser.add_argument('--output-dir', type=str, help='Directory for which to write the resulting dataset to disk as a HuggingFace dataset.')
    args = parser.parse_args()
    ds_dict = dict()
    jsonls = glob.glob(str(args.input_dir / '*.jsonl'))
    if len(jsonls) == 0:
        raise ValueError(f'Must provide a valid input directory! No jsonl files found under {args.input_dir}')
    for jsonl in jsonls:
        jsonl = Path(jsonl)
        if len(jsonls) == 1:
            # no splits!
            split = 'train'
        else:
            split = jsonl.stem
        if split not in ds_dict:
            dsdict_this_split = defaultdict(list)
        else:
            dsdict_this_split = ds_dict[split]
        total_this_file = get_num_examples(jsonl)
        pbar = tqdm(total=total_this_file, dynamic_ncols=True, desc=f'{ds_name} {split} split!')
        with jsonlines.open(jsonl) as f:
            for jsonobj in f:
                image_path = jsonobj['image']
                image_obj = Image.open(image_path)
                for key in required_fields:
                    if key == 'image':
                        dsdict_this_split[key].append(image_obj)
                    else:
                        dsdict_this_split[key].append(jsonobj[key])
                image_obj.close()
                pbar.update(1)
        pbar.close()
        if split not in ds_dict:
            ds_dict[split] = dsdict_this_split
    for split in ds_dict.keys():
        ds_dict[split] = Dataset.from_dict(ds_dict[split], split=split)
    ds = DatasetDict(ds_dict)
    ds.save_to_disk(args.output_dir)
    print('Done!')

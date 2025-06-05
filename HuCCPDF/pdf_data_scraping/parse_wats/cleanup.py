import os
import json
import shutil

def main(processed_jsonl_path, input_dir):
    with open(processed_jsonl_path, "r") as f:
        included_images = set()
        for line in f:
            pdf_data = json.loads(line)
            for image in pdf_data["images"]:
                if image:
                    included_images.add(image)
                    included_images.add(image.replace(".jpg", ".json").replace(".png", ".json").replace(".jpeg", ".json").replace(".webp", ".json"))
                    
    
    # remove images/jsons not in the processed_jsonl
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file not in included_images:
                os.remove(os.path.join(root, file))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--raw_jsonl_path", type=str)
    parser.add_argument("--processed_jsonl_path", type=str)
    args = parser.parse_args()
    main(args.processed_jsonl_path, args.input_dir)

import json
import argparse
import re
from urllib.parse import urlparse
from fastwarc.warc import ArchiveIterator, WarcRecordType
import simdjson
import time

def format_relative_to_absolute_path(page_url, relative_path):
    if relative_path.startswith("//"):
        abs_path = "http:" + relative_path
    else:
        if "./" in relative_path:
            relative_path = re.sub(r"\.+\/", "", relative_path)
        if not relative_path.startswith("/"):
            relative_path = "/" + relative_path
        domain_name = urlparse(page_url).netloc
        abs_path = "https://" + domain_name + relative_path
    return abs_path

URL_BAN_WORDS = ["porn", "xxx", "sex", "ad", "banner"]

video_ext_pattern = r'\.(mp4|wav|avi|mov|webm)$'
yt_pattern = r'https?://(www\.)?youtube\.com/watch\?v=\w+'
vimeo_pattern = r'https?://(www\.)?vimeo\.com/\d+'
doc_ext_pattern = r'\.(pdf)$'

def is_video_link(link):
    return re.search(yt_pattern, link.lower(), re.IGNORECASE) or re.search(vimeo_pattern, link, re.IGNORECASE) or re.search(video_ext_pattern, link.lower(), re.IGNORECASE)

def is_doc_link(link):
    return re.search(doc_ext_pattern, link.lower(), re.IGNORECASE)

def process_wat_file(wat_file_path, media_type, output_file):
    """
    Gets list of pdfs for jsonl file
    Args:
        wat_file_path: Local warc.wat.gz file path 
        media_type: Example "document"
        output_file: jsonl file
    """
    start = time.time()
    processed_links = set()
    
    with open(wat_file_path, "rb") as stream, open(output_file, "w") as f:
        for record in ArchiveIterator(stream, record_types=WarcRecordType.metadata, parse_http=False):
            record_data = simdjson.load(record.reader)
            envelope = record_data["Envelope"]
            payload = envelope["Payload-Metadata"]
            if "HTTP-Response-Metadata" not in payload:
                continue
            http_resp = payload["HTTP-Response-Metadata"]
            if "HTML-Metadata" not in http_resp:
                continue
            metadata = http_resp["HTML-Metadata"]
            record_url = record.headers["WARC-Target-URI"]

            if "Links" not in metadata:
                continue

            links = metadata["Links"]

            for link in links:
                if 'url' in link:
                    link = link['url']
                    
                    if any([word in link.lower() for word in URL_BAN_WORDS]) or link in processed_links:
                        continue
                                        
                    if (media_type == "video" and is_video_link(link)) or (media_type == "document" and is_doc_link(link)):
                        processed_links.add(link)
                        f.write(json.dumps({"url": format_relative_to_absolute_path(relative_path=link, page_url=record_url) if not link.startswith("http") else link}) + "\n")                        

    return time.time() - start

def log_results_to_wandb(result):
    """
    Callback function to log results to wandb.
    """
    if result:
        wandb.log(result)
        

def main():
    parser = argparse.ArgumentParser(
        description="Process WAT files and extract web documents."
    )
    parser.add_argument("--wat_file", help="Path to the WAT file")
    parser.add_argument("--media_type", type=str, choices=["video", "document"])
    parser.add_argument("--output_file", help="Path to the output JSONL file")
    parser.add_argument("--wandb", action="store_true", help="Log to wandb")
    parser.add_argument("--wandb_run_name", type=str)
    args = parser.parse_args()


    if args.wandb:
        wandb.init(project="interleaved-datasets", name=args.wandb_run_name)
        wandb.config.update(args)
        
    process_wat_file(args.wat_file, args.media_type, args.output_file)

    if args.wandb:
        wandb.finish()
        
        
if __name__ == "__main__":
    main()

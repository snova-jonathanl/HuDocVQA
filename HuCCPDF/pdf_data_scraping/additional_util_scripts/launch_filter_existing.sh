sntask run -j filter_all --timeout 1024:00:00 --cpus-per-task 16 --host-mem 100G --inherit-env \
--outfile /import/ml-sc-scratch3/nidhih/mm_hungarian/sambafeel/multimodal/pdf_data_scraping/outputs/filter1.log \
-- python run_filter_existing_pdfs.py --source_dir=/import/ml-sc-scratch3/nidhih/mm_hungarian/sambafeel/data/multimodal/pdf_data_scraping/outputs/single_warc_sntask/snapshots/out_jonathan_hungarian_ss0_CC-MAIN-2021-17/
    
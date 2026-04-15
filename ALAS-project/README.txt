This is an in-depth guide on how to run the code.

Because this project contains very heavy data files such as dataset of 5000 papers, and model checkpoints, these were omitted during file submission.

Thus, to run the code, they must be re-built. Please open, make any adjustments necessary, and run the code in the provided order below.

Phase 1 (phase1_gathering.Rmd): You should be able to run this code off-the-box as long as you have an R language interpreter. The csv metadata was provided so the code should skip over metadata gathering and go straight to pdf download. After downloading 1000 papers from each year (5000 papers), make sure you have the following file/folder structure: .\ALAS-project-main\ALAS-project\arxiv_database_csvs
Here, you should see the combined metadata csv file provided "arxiv_cs_2022_2026_raw_papers_combined" (if this file is missing, please re-download from GitHub) and you should see a folder called "all_years_pdfs" containing a folder for each year with pdfs like: arxiv_cs_2022_pdfs (folder) which inside has 1000 files (papers). If you see the individual years folder but no "all_years_pdfs", please create this folder manually and put the year_pdf folders inside it.

Phase 2 (phase2_preprocess.py): This code should be able to run as-is assuming everything in phase 1 went smoothly. This code will generate a really large file called "phase2_preprocessed.csv"

Phase 3 (phase3_training.py): As-is assuming same as before. This code takes as input phase2_preprocessed.csv (so any failure might be due to this file), and silverset_summaries.csv (provided, shouldn't cause failure, if it does, please just re-download the file).
First run: py phase3_training.py - this will create the synthetic training set required to train the model.
Then, you can run: py phase3_training.py --train - this will train the model and create the checkpoint configurations needed for inference.

Inference (inference_pipeline.py): Basic output debugging pipeline for the model, you may choose to run this or skip it as it does not create any meaningful files.

Model Evaluation (compare_models_seq2seq.py): This code assumes model checkpoints exists to run mode evaluations. If you stopped training early you may change the model name to your checkpoint on line 77. Otherwise, this code uses "phase3_outputs/silver_test_hybrid.csv" for evaluation which is provided so you can use even if training in phase 3 stopped early.
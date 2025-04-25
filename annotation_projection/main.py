import subprocess

def run_awesome_align():
    # === CONFIGURATION ===
    DATA_FILE = "/test/input.txt"
    MODEL_NAME_OR_PATH = "bert-base-multilingual-cased"
    OUTPUT_FILE = "/test/output.txt"
    BATCH_SIZE = "32"
    EXTRACTION = "softmax"

    # === RUN AWESOME-ALIGN ===
    command = [
        "CUDA_VISIBLE_DEVICES=0",
        "awesome-align",
        "--output_file", OUTPUT_FILE,
        "--model_name_or_path", MODEL_NAME_OR_PATH,
        "--tokenizer_name", MODEL_NAME_OR_PATH,
        "--data_file", DATA_FILE,
        "--extraction", EXTRACTION,
        "--batch_size", BATCH_SIZE
    ]

    # Use shell=True because of the CUDA env var prefix
    subprocess.run(" ".join(command), shell=True, check=True)
    print(f"Alignment completed! Output written to {OUTPUT_FILE}")




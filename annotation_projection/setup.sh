#!/bin/bash
echo "Cloning Awesome-Align repository..."
git clone https://github.com/neulab/awesome-align.git
cd awesome-align

echo "Installing Awesome-Align requirements..."
pip install -r requirements.txt
python setup.py install
cd ..

echo "Cloning IndicTrans2 repository..."
git clone https://github.com/AI4Bharat/IndicTrans2.git

echo "Installing core dependencies for IndicTrans..."
python3 -m pip install nltk sacremoses pandas regex mock "transformers>=4.33.2" mosestokenizer
python3 -c "import nltk; nltk.download('punkt')"

echo "Installing additional libraries for IndicTrans..."
python3 -m pip install bitsandbytes scipy accelerate datasets
python3 -m pip install sentencepiece

echo "Cloning IndicTransToolkit repository..."
git clone https://github.com/VarunGumma/IndicTransToolkit.git
cd IndicTransToolkit

echo "Installing IndicTransToolkit in editable mode..."
python3 -m pip install --editable ./
cd ..

echo "Setup complete!"
echo "Running Awesome-Align with configuration..."

# === CONFIGURATION ===
DATA_FILE="/test/input.txt"
MODEL_NAME_OR_PATH="bert-base-multilingual-cased"
OUTPUT_FILE="/test/output.txt"
BATCH_SIZE=32
EXTRACTION="softmax"

# === RUN AWESOME-ALIGN ===
CUDA_VISIBLE_DEVICES=0 awesome-align \
    --output_file "$OUTPUT_FILE" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --tokenizer_name "$MODEL_NAME_OR_PATH" \
    --data_file "$DATA_FILE" \
    --extraction "$EXTRACTION" \
    --batch_size "$BATCH_SIZE"

echo "🎉 Alignment completed! Output written to $OUTPUT_FILE"

#!/bin/bash
set -e  # Exit immediately on error
set -o pipefail
set -u  # Treat unset variables as errors

echo "[1/5] Installing core dependencies for IndicTrans..."
python3 -m pip install --upgrade pip
python3 -m pip install \
    nltk \
    sacremoses \
    pandas \
    regex \
    mock \
    "transformers>=4.33.2" \
    mosestokenizer

echo "[2/5] Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt')"

echo "[3/5] Installing additional libraries..."
python3 -m pip install \
    bitsandbytes \
    scipy \
    accelerate \
    datasets \
    sentencepiece

echo "[4/5] Cloning IndicTransToolkit repository..."
if [ ! -d "IndicTransToolkit" ]; then
    git clone https://github.com/VarunGumma/IndicTransToolkit.git
else
    echo "IndicTransToolkit directory already exists. Skipping clone."
fi

echo "[5/5] Installing IndicTransToolkit in editable mode..."
cd IndicTransToolkit
python3 -m pip install --editable .
cd ..

echo "✅ Setup complete!"

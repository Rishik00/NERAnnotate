#!/bin/bash
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

## These two were giving loads of trouble, so reinstalling them. 
python3 -m pip uninstall transformers -y
python3 -m pip uninstall numpy -y

python3 -m pip install transformers
python3 -m pip install numpy

echo "Setup complete!"



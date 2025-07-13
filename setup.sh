#!/bin/bash

# Create and activate virtual environment
python3 -m venv env
source env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Create data directory
mkdir -p data/valentini
cd data/valentini

# Download Valentini dataset
echo "Downloading Valentini dataset (clean + noisy)..."

wget https://datashare.ed.ac.uk/bitstream/handle/10283/1942/clean_trainset_28spk_wav.zip
wget https://datashare.ed.ac.uk/bitstream/handle/10283/1942/noisy_trainset_28spk_wav.zip
wget https://datashare.ed.ac.uk/bitstream/handle/10283/1942/clean_testset_wav.zip
wget https://datashare.ed.ac.uk/bitstream/handle/10283/1942/noisy_testset_wav.zip

# Unzip all files
unzip clean_trainset_28spk_wav.zip -d train/clean
unzip noisy_trainset_28spk_wav.zip -d train/noisy
unzip clean_testset_wav.zip -d test/clean
unzip noisy_testset_wav.zip -d test/noisy

# Remove zip files to save space
rm *.zip

echo "Valentini dataset is ready in: data/valentini"

cd ../../


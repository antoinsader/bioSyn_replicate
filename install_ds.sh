
set -e

# git clone https://github.com/antoinsader/bioSyn_replicate.git



# Define target directory
TARGET_DIR="data/data-ncbi-fair"

# Create target directory
mkdir -p "$TARGET_DIR"

# Unzip train_dict.zip into target directory
unzip -o train_dictionary.zip -d "$TARGET_DIR"

# Unzip traindev.zip into target directory
unzip -o traindev.zip -d "$TARGET_DIR"

echo "Files successfully extracted to $TARGET_DIR"

python -m venv myenv
source myenv/bin/activate

# === Step 5: Install dependencies ===
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu121 torch
pip install faiss-gpu-cu12
pip install tqdm transformers requests psutil

echo "Setup complete!"

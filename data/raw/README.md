# Raw data

Place the **MOROCO** corpus files here. They are intentionally **not** committed to git (see `.gitignore`).

## How to get the data

### Option A — Clone the official repo

```bash
git clone https://github.com/butnaruandrei/MOROCO.git _moroco_repo
# Then copy/move the relevant text + label files into this folder.
```

### Option B — Use the local zip you already have

Unzip it into this folder. After extraction you should see files like:

```
data/raw/
├── samples.txt
├── dialect_labels.txt
├── category_labels.txt
└── ...
```

(The exact filenames depend on which release you downloaded — adjust `src/preprocessing/load.py` accordingly.)

## After downloading

Run the inspection notebook to verify everything loads correctly:

```bash
jupyter lab notebooks/01_data_inspection.ipynb
```

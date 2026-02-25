# Brain Tumor Detection System

A simple Flask app to detect brain tumors from MRI images using a trained TensorFlow/Keras model.

## Files

- `app.py`: Flask application entrypoint.
- `models/model.h5`: Trained Keras model (stored with Git LFS).
- `templates/`: HTML templates used by the app.
- `uploads/`: Uploaded images directory.

## Setup

1. Create a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure Git LFS is installed to fetch the model file when cloning:

```bash
git lfs install
```

## Run

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser.

## Notes

- The `models/model.h5` file is tracked via Git LFS; cloning this repository requires Git LFS to download the large model file.
- The virtual environment directory is ignored by `.gitignore`.# Brain_tumor_detection_system
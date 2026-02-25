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
# Brain Tumor Detection System

Flask web application that classifies MRI images for brain tumor detection using a pre-trained TensorFlow/Keras model.

## Repository layout

- `app.py` — Flask application (server).
- `models/model.h5` — Trained Keras model (tracked with Git LFS).
- `templates/` — HTML templates.
- `uploads/` — Uploaded images (runtime).
- `requirements.txt` — Python dependencies.

## Requirements

- Python 3.8+ recommended
- Git and Git LFS to fetch large model file

## Setup (local development)

1. Clone the repository (Git LFS required to download the model):

```bash
git clone https://github.com/Rajatraiiii/Brain_tumor_detection_system.git
cd Brain_tumor_detection_system
git lfs install
git lfs pull
```

2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

4. (Optional) If you don't have `models/model.h5` locally, `git lfs pull` from step 1 will download it.

## Run

```bash
python app.py
```

Then open http://127.0.0.1:5000 in your browser. The app serves an upload page where you can submit MRI images for classification.

## Testing with sample images

Place sample MRI images in `uploads/` or use the web UI to upload images. The app will return a prediction (tumor / no tumor) and confidence score.

## Notes & best practices

- The `models/model.h5` file is tracked with Git LFS due to its size.
- Do not commit virtual environment folders; `.gitignore` already excludes common venv directories.
- If you need to retrain or update the model, create a new model file and track it with Git LFS.

## Contact

If you need help, open an issue on the GitHub repository or contact the project owner.
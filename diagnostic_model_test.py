from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np, os, sys

MODEL_PATH = 'models/model.h5'
CLASS_LABELS = ['pituitary','notumor','meningioma','glioma']
IMG_DIR = 'sample_MRI_Images'

print('Loading model:', MODEL_PATH)
model = load_model(MODEL_PATH)
print('Loaded model. output_shape =', getattr(model, 'output_shape', None))
try:
    last = model.layers[-1]
    print('Last layer:', type(last), getattr(last, 'activation', None))
except Exception:
    pass

if not os.path.exists(IMG_DIR):
    print('No image directory:', IMG_DIR)
    sys.exit(0)

files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.png','.jpg','.jpeg'))]
if not files:
    print('No images found in', IMG_DIR)
    sys.exit(0)

for f in files[:10]:
    path = os.path.join(IMG_DIR, f)
    try:
        img = load_img(path, target_size=(128,128))
        a = img_to_array(img)/255.0
        a = np.expand_dims(a,0)
        pred = model.predict(a)
        print('\nFile:', f)
        print('raw pred shape:', pred.shape)
        print('raw pred:', pred)
        arg = np.argmax(pred, axis=1)[0]
        conf = float(np.max(pred))
        label = CLASS_LABELS[arg] if arg < len(CLASS_LABELS) else f'idx_{arg}'
        print('argmax:', arg, 'label:', label, 'conf:', conf)
    except Exception as e:
        print('Error processing', f, e)

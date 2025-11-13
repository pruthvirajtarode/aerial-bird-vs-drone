# Helper to unzip provided datasets into project root. Run from project root.
import zipfile, os
for name in ['classification_dataset.zip','object_detection_Dataset.zip']:
    if os.path.exists(name):
        print('Extracting', name)
        with zipfile.ZipFile(name,'r') as z:
            z.extractall()
    else:
        print(name, 'not found in current folder. Place the zip files here if you want them extracted.')

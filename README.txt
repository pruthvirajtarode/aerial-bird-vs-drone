Aerial Bird vs Drone - Fixed Training Scripts

This ZIP contains two fixed Python scripts for training the classification model
on your local machine with TensorFlow 2.12 on Windows. They include workarounds
for the TF 2.12 JSON/EagerTensor logging bug.

Files:
- src/train_classifier.py
- src/model_utils.py
- README.txt

How to use
----------
1. Extract this ZIP into your project root:
   C:\Users\pruth\OneDrive\Desktop\final_internship_project_complete

2. Ensure your dataset is located at:
   classification_dataset/
       train/
           bird/
           drone/
       valid/
           bird/
           drone/

3. Activate your virtual environment:
   .\tf_env\Scripts\activate

4. Install required packages (if not already):
   pip install -r requirements.txt
   (or pip install tensorflow-cpu==2.12.0 gdown ultralytics)

5. Run training:
   python src\train_classifier.py --data classification_dataset --epochs 20 --batch 32

Notes
-----
- The scripts enforce eager execution and convert callback logs to floats to
  avoid the TF 2.12 serialization crash.
- If you still see errors, delete Python caches:
  Remove-Item -Force -Recurse src\__pycache__

# Expanded Demo Script (Detailed) - 2.5 minutes

0:00-0:10 - Intro
- "Hello, I'm Pruthviraj Tarode. This project detects birds vs drones in aerial imagery to improve airspace monitoring."

0:10-0:30 - Dataset overview
- "The classification dataset contains train/validation/test splits for Bird and Drone classes. The object detection dataset includes YOLOv8 labels for localization."

0:30-0:50 - Model & Approach
- "I used transfer learning with EfficientNetB0, with augmentations and early stopping. For explainability, Grad-CAM highlights model attention."

0:50-1:50 - Live demo (Streamlit)
- Open app, show header and UI.
- Upload a bird image, run Predict & Explain, show prediction + confidence, then Grad-CAM overlay.
- Repeat with drone image, emphasize differences in attention maps.

1:50-2:10 - Results
- Show confusion matrix and key metrics (accuracy/precision/recall). Explain failure cases briefly.

2:10-2:30 - Closing
- Mention improvements and invite questions.

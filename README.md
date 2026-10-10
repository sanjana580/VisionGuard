# 🔍 VisionGuard
### AI-Powered Industrial Surface Defect Classification

VisionGuard is a deep learning web application that classifies industrial steel surface defects using a ResNet18 model and transfer learning.

## 🌐 Live Demo

**[Click here to open VisionGuard](https://aakgrd3tezzatmdhaovad6.streamlit.app)**

## ✨ Features

- Upload an industrial surface image.
- Predict the defect category using deep learning.
- Display the prediction confidence and class probabilities.
- Explore descriptions of six defect categories.
- Access the application through a web browser.

## 🏭 Defect Categories

1. Crazing
2. Inclusion
3. Patches
4. Pitted Surface
5. Rolled-In Scale
6. Scratches

## 🛠️ Technologies Used

- Python
- PyTorch
- ResNet18
- Streamlit
- Pillow
- Git and GitHub

## 📊 Model Performance

The current experiment achieved **97.50% validation accuracy** on 360 validation images.

*Note: This is validation performance, not a guarantee of accuracy on unseen real-world industrial images.*

## 🚀 Run Locally

1. Clone this repository:

   ```bash
   git clone https://github.com/sanjana580/VisionGuard.git
   cd VisionGuard
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the application:

   ```bash
   python -m streamlit run app.py
   ```

## 🔮 Future Enhancements

- Independent testing on unseen datasets.
- Defect localization and segmentation.
- Evaluation on real-world industrial images.
- Inspection history and downloadable reports.

## 👩‍💻 Project

VisionGuard — Deep Learning-Based Industrial Surface Defect Classification.

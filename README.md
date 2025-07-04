## Skin Disease Prediction using ResNet152V2
A deep learning project that uses ResNet152V2 (transfer learning) to classify various skin diseases from image data. The project includes both a Streamlit-based interactive web app (main.py) and a Jupyter notebook (skin-disease-prediction.ipynb) for training and testing.

## 🔍 Objective
Early diagnosis of skin conditions can greatly improve treatment outcomes. This project provides an AI-powered tool to classify common skin infections from images using a fine-tuned ResNet152V2 model trained with transfer learning.

----
## Key Features
✅ Transfer Learning with ResNet152V2

🖼️ Upload skin images and get predictions instantly

📈 Evaluation via precision, recall, and F1-score

📦 Jupyter notebook for model training & experiments

🌐 Streamlit web app (main.py) for end-user interaction

 ## Model Performance
The model was evaluated on a test set of 229 images across 8 skin disease classes. Below is the classification report:

| **Class**                  | **Precision** | **Recall** | **F1-score** | **Support** |
| -------------------------- | ------------- | ---------- | ------------ | ----------- |
| BA-cellulitis              | 0.89          | 0.94       | 0.91         | 33          |
| BA-impetigo                | 1.00          | 0.95       | 0.97         | 20          |
| FU-athlete-foot            | 0.96          | 0.87       | 0.92         | 31          |
| FU-nail-fungus             | 0.94          | 0.97       | 0.95         | 32          |
| FU-ringworm                | 0.96          | 1.00       | 0.98         | 22          |
| PA-cutaneous-larva-migrans | 0.72          | 0.92       | 0.81         | 25          |
| VI-chickenpox              | 1.00          | 1.00       | 1.00         | 34          |
| VI-shingles                | 1.00          | 0.78       | 0.88         | 32          |

----------------------------------------------------------------------------------------
Overall Metrics:

🔹 Accuracy: 0.93 i.e **93%**

🔹 Macro Avg F1-Score: 0.93

🔹 Weighted Avg F1-Score: 0.93

-------
📁 Project Structure
```
Skin-Disease-Prediction/
├── LICENSE
├── README.md
├── main.py                         # Streamlit app
├── skin-disease-prediction.ipynb  # Jupyter notebook (training & analysis)
├── requirements.txt
```
---
## Dataset 
Download-->[Dataset](https://www.kaggle.com/datasets/subirbiswas19/skin-disease-dataset/data)

---
## My Kaggle Notebook Link
## [Kaggle](https://www.kaggle.com/code/fall2fire/skin-disease-prediction)

---
## Getting Started
1. Clone the Repository
```
git clone https://github.com/Pranav-Uniyal/Skin-Disease-Prediction.git
cd Skin-Disease-Prediction
```
2. Install Dependencies
```
pip install -r requirements.txt
```
3. Launch Streamlit App
```
streamlit run main.py
```
----
## Model Details
🔹 Base: ResNet152V2 (pretrained on ImageNet)

🔹 Added Layers: GlobalAveragePooling → Dropout → Dense (Softmax)

🔹 Loss: categorical_crossentropy

🔹 Optimizer: Adam

🔹 Image Input Size: 224x224x3

----
## Screenshot
![image](https://github.com/user-attachments/assets/b68b38de-c07a-4979-bc94-eaebe07e8848)
![image](https://github.com/user-attachments/assets/ccf97c98-0b03-449a-aec3-69018c664fd0)

----
## Future Scope
Add Grad-CAM visual explanations for prediction insights

Expand to additional skin diseases and rare infections

Deploy live on Streamlit Cloud or Hugging Face Spaces

## 📃 License
This repository is licensed under the MIT License.

## Developed By
Pranav Uniyal
[Github](https://github.com/Pranav-Uniyal) [Kaggle](https://www.kaggle.com/fall2fire)

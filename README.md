\# 👁️ Retinal Disease Detection using OCT Images



An end-to-end deep learning system for automated retinal disease classification from Optical Coherence Tomography (OCT) images, with explainable AI and a web-based inference portal.



---



\## 🔬 Diseases Classified

\- CNV (Choroidal Neovascularization)

\- DME (Diabetic Macular Edema)

\- DRUSEN

\- NORMAL



---



\## 🌐 Web Application Demo

The model is deployed locally using Streamlit, allowing users to upload OCT images and receive predictions with confidence scores and Grad-CAM visual explanations.



!\[Web App Demo](https://github.com/<rayankhanongit>/<retinal-disease-detector-oct>/raw/main/screenshots/web\_app\_gradcam.png)



---



\## 📊 Dataset

\- \*\*OCT2017 dataset\*\*

\- ~84,000 grayscale retinal OCT images

\- Train / Validation / Test split

\- 4-class classification problem



> Dataset is not included due to size and licensing constraints.



---



\## 🧠 Model Architecture

\- ResNet-18 (pretrained on ImageNet)

\- Modified input layer for grayscale OCT images

\- Transfer learning using PyTorch



---



\## ⚙️ Training Setup

\- Framework: PyTorch

\- Hardware: NVIDIA RTX 3050 (CUDA enabled)

\- Loss: CrossEntropyLoss

\- Optimizer: Adam

\- Batch size: 16



---



\## 📈 Results



\### Classification Performance

!\[Evaluation Results](screenshots/evaluation\_results.png)



\- Test Accuracy: \*\*~99%\*\*

\- Strong precision, recall, and F1-score across all classes

\- Minimal class confusion



---



\## 🔍 Explainability (Grad-CAM)

Grad-CAM was applied to visualize model attention.  

Heatmaps confirm the model focuses on \*\*clinically relevant retinal layers\*\*, validating trustworthy predictions.



---



\## 🗂️ Project Structure

retinal-oct-project/

├── src/

│ ├── data/

│ │ └── dataset.py

│ ├── models/

│ │ └── resnet\_model.py

│ ├── train.py

│ ├── evaluate.py

│ └── gradcam\_utils.py

├── app.py

├── experiments/

├── screenshots/

├── README.md

└── .gitignore





---



\## ▶️ How to Run



```bash

\# Train the model

python -m src.train



\# Evaluate

python -m src.evaluate



\# Run web app

streamlit run app.py



⚠️ Disclaimer



This project is intended for educational and research purposes only and is not approved for clinical diagnosis.





Save and close.



---



\# 🧪 PART 4: FINAL CHECK BEFORE GITHUB PUSH



Run these checks:



```powershell

git status


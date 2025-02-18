# Breast Cancer Prediction using Logistic Regression  

## 📌 Project Overview  
This project builds a **Breast Cancer Prediction Model** using **Logistic Regression** on the  
[Breast Cancer Wisconsin Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data).  
Hyperparameter tuning with **GridSearchCV** improved accuracy from **98.2% to 99.4%**.  

## 📂 Dataset  
- **Source:** [Breast Cancer Wisconsin Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)  
- **Description:** The dataset contains **569** observations with **30 numerical features** extracted from digitized images  
  of fine needle aspirate (FNA) of breast masses.  
- **Target Variable:**  
  - **Malignant (1)** – Cancerous tumor  
  - **Benign (0)** – Non-cancerous tumor  

## 🚀 Technologies Used  
- **Python**  
- **Pandas & NumPy** – Data handling and preprocessing  
- **Scikit-Learn** – Machine Learning (Logistic Regression, GridSearchCV)  
- **Matplotlib & Seaborn** – Data visualization  

## ⚡ Model Performance  
| Model | Accuracy |
|--------|---------|
| Logistic Regression (Default) | 98.2% |
| Logistic Regression (GridSearchCV) | 99.4% |

## 🛠️ Installation & Usage  
### **1️⃣ Install Dependencies**  
Ensure you have Python installed, then install the required libraries:  
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### **2️⃣ Download Dataset**  
Download the dataset from Kaggle and place it in the project directory.  

### **3️⃣ Run the Model**  
Execute the ipynb file to train and evaluate the model

## 📜 Project Workflow  
### **1️⃣ Data Preprocessing**  
✅ Removed unnecessary columns (`id`, unnamed column).  
✅ Converted **diagnosis** column (`M → 1, B → 0`).  
✅ Scaled the features using **StandardScaler**.  
✅ Split the dataset into **70% training, 30% testing**.  

### **2️⃣ Model Training & Evaluation**  
✔ **Baseline Logistic Regression Model** achieved **98.2% accuracy**.  
✔ **Hyperparameter tuning** using **GridSearchCV** (5-fold cross-validation).  
✔ **Best Model Achieved** → **99.4% accuracy**.  

### **3️⃣ Final Model Testing**  
✔ Predictions made on test data using the best model.  
✔ Evaluated using **accuracy_score, classification_report, confusion_matrix**.  

### **📊 Classification Report **  

- **Precision**: Measures how many of the predicted positive cases were actually correct.  
- **Recall**: Measures how many actual positive cases were correctly identified.  
- **F1-score**: A balance between precision and recall.  
- **Support**: The number of actual occurrences of each class in the dataset.  

#### **Results Summary:**  
✅ **Benign Tumors (0)** → **99% precision, 100% recall** → Almost all benign cases were correctly classified.  
✅ **Malignant Tumors (1)** → **100% precision, 98% recall** → One malignant case was misclassified as benign.  
✅ **Overall Accuracy:** **99%** → The model performs very well in predicting breast cancer.  

## 🔍 Confusion Matrix  

- **108** True Positives (Benign correctly classified).  
- **62** True Negatives (Malignant correctly classified).  
- **1** False Negative (Malignant misclassified as Benign).  
- **0** False Positives (No Benign misclassified as Malignant).  


## 📜 License  
This project is open-source and available for personal and educational use.  

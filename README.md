# **End-to-End Logistic Regression - Loan Approval Prediction(1: Approved, 0: Rejected)**  

## **Problem Statement**  
This project aims to predict **loan approval** status based on applicant details using **Logistic Regression**. The dataset contains various attributes such as income, credit history, loan amount, and more. We apply feature engineering, preprocessing, and machine learning techniques to build an optimal classification model.

---

## **Getting Started**  

### **Install Dependencies**  
Before running the code, install all required dependencies using:
```bash
pip install -r requirements.txt
```
---

### **Repository Structure**

```plaintext
END-END-LOGISTIC-REGRESSION/
│── notebook/                     # Jupyter notebooks for step-by-step implementation
│   ├── loan_approval_model.ipynb      # Training notebook
│   ├── loan_approval_prediction.ipynb # Prediction notebook
│   ├── final_model_lr.joblib          # Saved Logistic Regression model
│   ├── scaler.joblib                  # Standardization scaler
│   ├── feature_list.json              # Selected features for modeling
│
│── output/                       # Predictions on new unseen data
│   ├── predictions.csv
│
│── processed_data/                # Preprocessed data before modeling
│   ├── processed_data.csv
│
│── raw/                           # Raw dataset files
│   ├── loan_data.csv
│   ├── new_loan_data.csv
│
│── saved_models/                   # Trained models and transformations
│   ├── final_model_lr.joblib
│   ├── scaler.joblib
│
│── src/                            # Modular Python scripts for production-ready use
│   ├── data_preprocessing.py        # Handles missing values, outliers, transformations
│   ├── load_configuration.py        # Loads config.yaml settings
│   ├── model_build.py               # Builds and trains the classification model
│   ├── prediction.py                # Generates predictions on new data
│ 
├── config.yaml                      # Configuration file for easy parameter changes
├── main.py                          # **Main script to execute entire pipeline**  
│── README.md                        # This file  
│── requirements.txt                 # Required dependencies

```
---
### **How to Use**

### Option 1: Running Jupyter Notebooks (Independent of `src/`)  
If you prefer a step-by-step interactive approach, use the Jupyter Notebooks inside the `notebook/` folder.  

1. **Train the Model**  
   - Open and run [`loan_approval_model.ipynb`](notebook/loan_approval_model.ipynb) to train the model.  

2. **Make Predictions**  
   - Open and run [`loan_approval_prediction.ipynb`](notebook/loan_approval_prediction.ipynb) to make predictions on new data.  

**Note:** All generated outputs will be saved within the same `notebook/` folder.

### Option 2: Running the Modular Pipeline (Production Ready)  

For a structured, modular approach, execute the pipeline using:  

```bash
python main.py

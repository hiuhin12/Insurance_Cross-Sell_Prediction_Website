# Vehicle Insurance Cross-Sell Prediction with Machine Learning and Customer Segmentation 🚗📊

## Introduction
This project focuses on predicting vehicle insurance cross-sell opportunities using advanced machine learning models. The goal is to identify potential customers for vehicle insurance based on existing health insurance data, utilizing classification algorithms and customer segmentation techniques to enhance sales strategies.

## Contents

1. **Source Code**
   - **Python**: The core machine learning models and data processing scripts are implemented in Python, utilizing libraries such as Scikit-Learn, CatBoost, and Flask.
     - **Files:**
       - `app.py`: The main backend application built using Flask for handling customer input, prediction, and segmentation.
       - `data_preprocessing.py`: Preprocessing pipeline for cleaning and transforming data before model inference.
       - `trained_catboost_classifier.pkl`:  Pretrained CatBoost model used for customer segmentation.
    
2. **Website**
   - `amain.html`: Main homepage of the insurance prediction system.
   - `enterinfor.html`: Form for entering customer details manually.
   - `enterinfor-result.html`: Page displaying the prediction results.
   - `upload.html`: Page for batch uploading customer data.
   - `upload-result.html`: Page displaying batch processing results.
       
3. **Machine Learning Models**
   - **Voting Classifier Model:** Predicts customer interest in vehicle insurance.
   - **CatBoost Classifier Model:** Segments customers into relevant groups based on demographic and behavioral data.

4. **Datasets**
   - **Data Source:**
     - Health Insurance Cross Sell Prediction 🏠 🏥 from [Kaggle.com](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction).
     - **Files:**
         - `customer_data.csv`: Contains preprocessed customer attributes for training and testing the models. ​
         - Features include Gender, Age, Driving License, Region Code, Previously Insured, Vehicle Age, Vehicle Damage, Annual Premium, Policy Sales Channel, Vintage, Response.

## How It Works

1. **Enter Customer Data**
   - Manually input customer information through the web form.
   - Upload a CSV file containing customer data for bulk predictions.

2. **Prediction & Segmentation**
   - The Voting Classifier predicts the likelihood of a customer purchasing vehicle insurance.
   - The CatBoost Classifier segments customers into groups for targeted marketing strategies.
  
3. **Results Displayed**
   - The website provides instant feedback on customer predictions and segmentation results.
  
## Deployment
   - **Flask-based Backend:** The website provides instant feedback on customer predictions and segmentation results.
   - **Frontend:** Static web pages with dynamic content rendering.
   - **Data Processing:** Implemented with `data_preprocessing.py` to handle missing values, scaling, and feature engineering.
   
## Further Information

**Contact Person for Questions:**  
   - Name: Vo Minh Thanh 
   - Email: thanhvm21416c@st.uel.edu.vn

**Include Credits**
   - Leader: Vo Minh Thanh
   - Member:
     + Tran Anh Khoa
     + Nguyen Trung Hieu Hien
     + Nguyen Quoc Huy
     + Huynh Thi Thanh Truc

**Keywords:** Insurance, Machine Learning, Customer Segmentation, Cross-Selling, Vehicle Insurance, Predictive Analytics

**Language:** English

**Date of Data Collection:** September, 2023

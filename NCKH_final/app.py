import csv
import os
import numpy as np
import pickle
import pandas as pd
from flask import Flask, redirect, render_template, request, url_for
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from data_preprocessing import DataPreprocessing

app = Flask(__name__)

model = {
    'voting_classifier': pickle.load(open(r'C:\Users\Admin\OneDrive - uel.edu.vn\Documents\GitHub\Funny-Coder\NCKH_final\trained_voting_classifier.pkl', 'rb')),
    'catboost_classifier': pickle.load(open(r'C:\Users\Admin\OneDrive - uel.edu.vn\Documents\GitHub\Funny-Coder\NCKH_final\trained_catboost_classifier.pkl', 'rb'))
}

@app.route('/amain')
def amain():
    return render_template('amain.html')

@app.route('/')
def home():
    return render_template('amain.html')

@app.route('/enter-information', methods=['POST', 'GET'])
def enterinformation():
    if request.method == 'POST':
        # Logic để xử lý dữ liệu form gửi lên
        # Ví dụ: Lấy dữ liệu từ form, tiến hành dự đoán, v.v...
        return redirect(url_for('enterinformation'))
    # Nếu là GET, chỉ cần hiển thị trang/form
    return render_template('enterinfor.html')

@app.route('/enter-information/result', methods=['POST'])
def predict():
    if request.method == 'POST':

        id = int(request.form.get('id'))
        Gender = int(request.form.get('Gender'))
        Age = int(request.form.get('Age'))
        Driving_License = int(request.form.get('Driving_License'))
        Region_Code = int(request.form.get('Region_Code'))
        Previously_Insured = int(request.form.get('Previously_Insured'))
        Vehicle_Age = int(request.form.get('Vehicle_Age'))
        Vehicle_Damage = int(request.form.get('Vehicle_Damage'))
        Annual_Premium = float(request.form.get('Annual_Premium'))
        Policy_Sales_Channel = int(request.form.get('Policy_Sales_Channel'))
        Vintage = float(request.form.get('Vintage'))

        csv_file = 'customer_data.csv'
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([id, Gender, Age, Driving_License, Region_Code, Previously_Insured, Vehicle_Age, Vehicle_Damage, Annual_Premium, Policy_Sales_Channel, Vintage])

        X = np.array([[id, Gender, Age, Driving_License, Region_Code, Previously_Insured, Vehicle_Age, Vehicle_Damage, Annual_Premium, Policy_Sales_Channel, Vintage]])
        
        column_names = ['id', 'Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']
        X_df = pd.DataFrame(X, columns=column_names)

        preprocessor = DataPreprocessing()
        preprocessor.fit(X_df)
    
        voting_classifier = model['voting_classifier']
        prediction_result = voting_classifier.predict(X_df)

        catboost_classifier = model['catboost_classifier']
        segmentation_result = catboost_classifier.predict(X_df)

        return render_template('enterinfor-result.html', prediction=prediction_result[0], segmentation=segmentation_result[0])
    return redirect(url_for('home'))

@app.route('/upload-data', methods=['POST', 'GET'])
def uploaddata():
    if request.method == 'POST':
        # Logic để xử lý dữ liệu form gửi lên
        # Ví dụ: Lấy dữ liệu từ form, tiến hành dự đoán, v.v...
        return redirect(url_for('uploaddata'))
    # Nếu là GET, chỉ cần hiển thị trang/form
    return render_template('upload.html')

@app.route('/upload-data/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        # Lưu file vào thư mục mong muốn
        filepath = os.path.join(r'C:\Users\Admin\OneDrive - uel.edu.vn\Documents\GitHub\Funny-Coder\NCKH_final\uploads', file.filename)
        file.save(filepath)

        # Đọc dữ liệu từ file vừa được upload
        dataset = pd.read_csv(filepath)

        preprocessor = DataPreprocessing()
        preprocessor.fit(dataset)
        features_df = preprocessor.transform(dataset)

        voting_classifier = model['voting_classifier']
        predictions = voting_classifier.predict(features_df)

        castboost_classifier = model['catboost_classifier']
        classifications = castboost_classifier.predict(features_df)
        
        dataset['Prediction'] = predictions
        dataset['Classification'] = classifications[:,0]

        data_records = dataset.to_dict(orient='records')

        return render_template('upload-result.html', records=data_records)
    return redirect(url_for('home'))
    
if __name__ == '__main__':
    app.run(debug=True)
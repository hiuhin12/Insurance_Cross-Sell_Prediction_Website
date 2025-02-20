"""# III. DATA PREPROCESSING"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
class DataPreprocessing(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scalers = {}
        self.policy_channel_mapping = None
        self.region_mapping = None
        self.upper_whisker = None

    def fit(self, X, y=None):
        # Xác định và lưu giới hạn trên cho 'Annual_Premium'
        Q1 = X['Annual_Premium'].quantile(0.25)
        Q3 = X['Annual_Premium'].quantile(0.75)
        IQR = Q3 - Q1
        self.upper_whisker = Q3 + 1.5 * IQR

        # Fit MinMaxScaler cho 'Annual_Premium' và 'Vintage'
        self.scalers['Annual_Premium'] = MinMaxScaler().fit(X[['Annual_Premium']])
        self.scalers['Vintage'] = MinMaxScaler().fit(X[['Vintage']])

        # Lưu các giá trị cho 'Policy_Sales_Channel' và 'Region_Code'
        policy_channel_counts = X['Policy_Sales_Channel'].value_counts()
        self.policy_channel_mapping = {k: 'Channel_A' if v > 100000 else 'Channel_B' if 74000 < v <= 100000 else 'Channel_C' if 10000 < v <= 74000 else 'Channel_D' for k, v in policy_channel_counts.items()}

        region_counts = X['Region_Code'].value_counts()
        self.region_mapping = {k: 'Region_A' if v >= 100000 else 'Region_B' if 11000 < v < 100000 else 'Region_C' for k, v in region_counts.items()}

        return self

    def transform(self, X, y=None):
        df = X.copy()

        # Xử lý ngoại lệ cho 'Annual_Premium'
        df['Annual_Premium'] = np.where(df['Annual_Premium'] > self.upper_whisker, self.upper_whisker, df['Annual_Premium'])

        # Chuyển đổi thuộc tính phân loại
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
        df['Vehicle_Age'] = df['Vehicle_Age'].map({'< 1 Year': 1, '1-2 Year': 2, '> 2 Years': 3})
        df['Vehicle_Damage'] = df['Vehicle_Damage'].map({'Yes': 1, 'No': 0})

        # Áp dụng MinMaxScaler
        for feature, scaler in self.scalers.items():
            df[feature] = scaler.transform(df[[feature]])

        # Chuyển đổi 'Age', 'Policy_Sales_Channel', và 'Region_Code'
        df['Age'] = df['Age'].apply(lambda x: 1 if x >= 20 and x <= 34 else 2 if x > 34 and x <= 61 else 3)
        df['Policy_Sales_Channel'] = df['Policy_Sales_Channel'].map(lambda x: self.policy_channel_mapping.get(x, 'Channel_D')).map({'Channel_A': 1, 'Channel_B': 2, 'Channel_C': 3, 'Channel_D': 4})
        df['Region_Code'] = df['Region_Code'].map(lambda x: self.region_mapping.get(x, 'Region_C')).map({'Region_A': 1, 'Region_B': 2, 'Region_C': 3})

        return df
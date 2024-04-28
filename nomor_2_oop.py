import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np
from sklearn.model_selection import GridSearchCV
import pickle
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, RobustScaler

# PROSES TRAINING MODEL XGBOOST

class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_df = None
    def load_data(self):
        self.data = pd.read_csv(self.file_path)
    def create_input_output(self, target_column):
        self.output_df = self.data[target_column]
        self.input_df = self.data.drop(target_column, axis=1)

class ModelHandler:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.createModel()
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_predict = [None] * 5

    # The model that I will use is XGBoost
    def createModel(self,random_state=0,n_estimators=100,criterion='squared_error'):
        self.model = xgb.XGBClassifier()
    
    def fill_na(self):
        self.x_train['CreditScore'] = self.x_train['CreditScore'].fillna(value=self.x_train['CreditScore'].mean())
        # We use x_train mean to impute value in test data
        self.x_test['CreditScore'] = self.x_test['CreditScore'].fillna(value=self.x_train['CreditScore'].mean())

    def drop_column(self):
        self.x_train = self.x_train.drop(columns=['Unnamed: 0','CustomerId','id','Surname'])
        self.x_test = self.x_test.drop(columns=['Unnamed: 0','CustomerId','id','Surname'])

    def handle_balance_variable(self):
        self.x_train['Balance'] = self.x_train['Balance'].replace(0, self.x_train['Balance'].mean())
        self.x_test['Balance'] = self.x_test['Balance'].replace(0, self.x_train['Balance'].mean())

    def split_data(self):
        split_index = int(len(self.input_data) * 0.8)  # 80% training data, 20% testing data
        self.x_train = self.input_data.iloc[:split_index]
        self.x_test = self.input_data.iloc[split_index:]
        self.y_train = self.output_data.iloc[:split_index]
        self.y_test = self.output_data.iloc[split_index:]
    
    def scaling(self):
        scaler1 = StandardScaler()
        scaler2 = RobustScaler()

        column_standard = ['Balance','EstimatedSalary']
        column_robust = ['CreditScore','Age']

        self.x_train[column_standard] = scaler1.fit_transform(self.x_train[column_standard])
        self.x_test[column_standard] = scaler1.transform(self.x_test[column_standard])

        self.x_train[column_robust] = scaler2.fit_transform(self.x_train[column_robust])
        self.x_test[column_robust] = scaler2.transform(self.x_test[column_robust])
    
    def encoding(self):
        mapping = {"Geography":{"Spain":0,"France":2,"Germany":3}, "Gender":{"Female":0,"Male":1}}
        self.x_train = self.x_train.replace(mapping)
        self.x_test= self.x_test.replace(mapping)
    
    def handle_imbalanced(self):
        os = SMOTE(random_state=0)
        x_train_resampled, y_train_resampled = os.fit_resample(self.x_train, self.y_train)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_train_resampled, y_train_resampled, test_size = 0.2, random_state = 42)

    # TRAIN AND EVALUATE MODEL
    def train_model(self):
        self.model.fit(self.x_train, self.y_train)
    
    def evaluate_model(self):
        predictions = self.model.predict(self.x_test)
        acc = accuracy_score(self.y_test, predictions)
        precision = precision_score(self.y_test, predictions)
        recall = recall_score(self.y_test, predictions)
        f1 = f1_score(self.y_test, predictions)
        print(f"Accuracy score: {acc}")
        print(f"Precision score: {precision}")
        print(f"Recall score: {recall}")
        print(f"F1 score: {f1}")
    
    def classification_report_print(self):
        print('\nClassification Report\n')
        print(classification_report(self.y_test, self.model.predict(self.x_test)))
    
    def makePrediction(self):
        self.y_predict = self.model.predict(self.x_test)
    
    def tuningParameter(self):
        parameters = {
                'max_depth': [3, 5, 7],
                'min_child_weight': [1, 3, 5],
                'gamma': [0, 0.1, 0.2],
                'learning_rate': [0.01, 0.1, 0.2],
                'n_estimators': [100, 200, 300]
        }
        xgboost = xgb.XGBClassifier()
        xgboost = GridSearchCV(xgboost ,
                param_grid = parameters,
                scoring='accuracy',
                cv=5
                )
        
        xgboost.fit(self.x_train, self.y_train)
        print("Tuned Hyperparameters:", xgboost.best_params_)
        print("Accuract=y score:", xgboost.best_score_)
        self.createModel(random_state = 0, max_depth = xgboost.best_params_['max_depth'],
                         min_child_weight = xgboost.best_params_['min_child_weight'], gamma = xgboost.best_params_['gamma'],
                         learning_rate = xgboost.best_params_['learning_rate'], n_estimators = xgboost.best_params_['n_estimators'] )
        
    def save_model_to_file(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)
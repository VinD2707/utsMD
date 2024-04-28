import streamlit as st
import joblib
import numpy as np
import os
# from sklearn.preprocessing import StandardScaler

# Load the machine learning model
model = joblib.load('xgb_model.pickle')

def main():
    st.title('Machine Learning Model Deployment')

    # Add user input components for 4 features
    Creditscore = st.slider('CreditScore', min_value=0.0, max_value=860.0, value=0.5)
    Age = st.slider('Age', min_value=0.0, max_value=95.0, value=1.0)
    Balance = st.slider('Balance', min_value=0.0, max_value=250000.0, value=0.5)
    EstimatedSalary = st.slider('EstimatedSalary', min_value=0.0, max_value=200000.0, value=0.5)
    
    # Mapping dictionaries for categorical variables
    geography_mapping = {'Spain': 0, 'France': 1, 'Germany': 2}
    gender_mapping = {'Female': 0, 'Male': 1}

    # Add a dropdown menu for categorical prediction
    Geography = ['Spain', 'France', 'Germany']
    selected_category1 = st.selectbox('Select Geography', Geography)
    
    Gender = ['Female', 'Male']
    selected_category2 = st.selectbox('Select Gender', Gender)

    # Convert selected categories to numerical values
    selected_category1_numeric = geography_mapping[selected_category1]
    selected_category2_numeric = gender_mapping[selected_category2]

    # Display the selected numerical values
    # st.write('Selected Geography (Numeric):', selected_category1_numeric)
    # st.write('Selected Gender (Numeric):', selected_category2_numeric)

    Tenure = ['0','1','2','3','4','5','6','7','8','9','10']
    selected_category3 = st.selectbox('Select Tenure', Tenure)

    NumOfProducts = ['1','2','3','4']
    selected_category4 = st.selectbox('Select NumOfProducts', NumOfProducts)

    HasCrCard = ['0','1']
    selected_category5 = st.selectbox('Select HasCrCard', HasCrCard)

    IsActiveMember = ['0','1']
    selected_category6 = st.selectbox('Select IsActiveMember', IsActiveMember)

    if st.button('Make Prediction'):
        features = [Creditscore,Age,Balance,EstimatedSalary,selected_category1_numeric,selected_category2_numeric,selected_category3,selected_category4
                    ,selected_category5, selected_category6]
        result = make_prediction(features)
        st.success(f'The prediction is: {result}')

def make_prediction(features):
    # Use the loaded model to make predictions
    # Replace this with the actual code for your model
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]


if __name__ == '__main__':
    main()

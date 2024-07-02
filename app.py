import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import shap
from data_preprocessing import preprocess_data
from model_utils import load_model, plot_roc_curve, plot_feature_importance, plot_shap_summary

# Load the model
rf_model = load_model('randomforest_model.pkl')

# Function to load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv('mfa_factor_aoa.csv')
    return preprocess_data(data)

# Main Streamlit app
def main():
    st.title('Credit Scoring Model Demo')

    # Load data
    data = load_data()

    # Sidebar for user input
    st.sidebar.header('User Input Features')
    
    user_input = {}
    for feature in rf_model.feature_names_in_:
        user_input[feature] = st.sidebar.slider(
            feature, 
            float(data[feature].min()), 
            float(data[feature].max()), 
            float(data[feature].mean())
        )

    # Create a DataFrame for the user input
    user_input_df = pd.DataFrame(user_input, index=[0])

    # Main page
    st.subheader('User Input features')
    st.write(user_input_df)

    # Make prediction
    prediction = rf_model.predict_proba(user_input_df)
    st.subheader('Prediction')
    st.write(f'Probability of Bad: {prediction[0][1]:.2f}')

    # Feature Importance
    st.subheader('Feature Importance')
    fig_importance = plot_feature_importance(rf_model)
    st.pyplot(fig_importance)

    # ROC Curve
    st.subheader('ROC Curve')
    X = data.drop(['Bad', 'decision_mth2'], axis=1)
    y = data['Bad']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    y_pred = rf_model.predict_proba(X_test)[:, 1]
    fig_roc = plot_roc_curve(y_test, y_pred)
    st.pyplot(fig_roc)

    # SHAP Values
    st.subheader('SHAP Values')
    fig_shap = plot_shap_summary(rf_model, X_test)
    st.pyplot(fig_shap)

if __name__ == '__main__':
    main()

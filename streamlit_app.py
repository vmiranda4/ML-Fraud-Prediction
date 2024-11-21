import streamlit as st
import numpy as np
import pandas as pd
import joblib  # For model loading

# Load the trained model
model = joblib.load('trained_model.pkl')

# Load the imputer
imputer = joblib.load('knn_imputer.pkl')

# Load the scalers
robust_scaler = joblib.load('robust_scaler.pkl')
quantile_transformer = joblib.load('quantile_transformer.pkl')


# Preprocessing function
def preprocess_input(raw_data):
    """Apply all necessary transformations to raw data."""

    # Ensure raw_data contains necessary keys
    processed_data = {}

    if raw_data['C'] == None:
        data_for_imputation = pd.DataFrame([{
        'C_filled': raw_data['C'],
        'A': raw_data['A'],
        'Monto': raw_data['Monto'],
        'S': raw_data['S']
        }])

        # Impute missing values
        imputed_data = imputer.transform(data_for_imputation)

        # Update the dictionary with imputed values
        raw_data['C_filled'], raw_data['A'], raw_data['Monto'], raw_data['S'] = imputed_data[0]

    else:
        raw_data['C_filled'] = raw_data['C']
    
    # Transform features
    processed_data['B'] = raw_data['B']
    processed_data['S_robust_scaled'] = robust_scaler.transform([[raw_data['S']]])[0][0]
    processed_data['P_robust_scaled'] = robust_scaler.transform([[raw_data['P']]])[0][0]
    processed_data['C_filled_robust_scaled'] = robust_scaler.transform([[raw_data['C_filled']]])[0][0]
    
    # Log-transformed Features
    for col in ['M', 'Q', 'N', 'H', 'D', 'A', 'R', 'O']:
        processed_data[f'{col}_log'] = np.log1p(raw_data[col] + 1)
    
    # Null Indicator Features
    processed_data['K_is_null'] = 1 if pd.isnull(raw_data['K']) else 0
    processed_data['C_is_null'] = 1 if pd.isnull(raw_data['C']) else 0
    
    # Quantile Transformed Feature
    processed_data['Monto_quantile_transformed'] = quantile_transformer.transform([[raw_data['Monto']]])[0][0]
    
    # One-Hot Encoded Features
    country_list = [
        'AR', 'AU', 'BR', 'CA', 'CH', 'CL', 'CO', 'ES', 'FR', 'GB', 
        'GT', 'IT', 'KR', 'MX', 'PT', 'TR', 'UA', 'US', 'UY'
    ]
    for country in country_list:
        processed_data[f'country_{country}'] = 1 if raw_data['J'] == country else 0

    # Ensure correct order of variables
    ordered_data = pd.DataFrame([[processed_data[col] for col in [
        'B', 'S_robust_scaled', 'P_robust_scaled', 'M_log', 'K_is_null',
        'C_is_null', 'Q_log', 'N_log', 'H_log', 'Monto_quantile_transformed',
        'D_log', 'A_log', 'C_filled_robust_scaled', 'R_log', 'O_log',
        'country_AR', 'country_AU', 'country_BR', 'country_CA', 'country_CH',
        'country_CL', 'country_CO', 'country_ES', 'country_FR', 'country_GB',
        'country_GT', 'country_IT', 'country_KR', 'country_MX', 'country_PT',
        'country_TR', 'country_UA', 'country_US', 'country_UY'
    ]]], columns=[
        'B', 'S_robust_scaled', 'P_robust_scaled', 'M_log', 'K_is_null',
        'C_is_null', 'Q_log', 'N_log', 'H_log', 'Monto_quantile_transformed',
        'D_log', 'A_log', 'C_filled_robust_scaled', 'R_log', 'O_log',
        'country_AR', 'country_AU', 'country_BR', 'country_CA', 'country_CH',
        'country_CL', 'country_CO', 'country_ES', 'country_FR', 'country_GB',
        'country_GT', 'country_IT', 'country_KR', 'country_MX', 'country_PT',
        'country_TR', 'country_UA', 'country_US', 'country_UY'
    ])
    
    return ordered_data

# Streamlit App
st.title("Fraud Prediction App")

# Input Fields
st.subheader("Enter Data")
A = st.number_input("A")
B = st.number_input("B")
C_null = st.checkbox("Is C Null?", value=False)
C = None if C_null else st.number_input("C", value=0.0)
D = st.number_input("D")
E = st.number_input("E")
F = st.number_input("F")
G = st.number_input("G")
H = st.number_input("H")
I = st.number_input("I")
J = st.selectbox("Country (J)", [
    'AR', 'AU', 'BR', 'CA', 'CH', 'CL', 'CO', 'ES', 'FR',
    'GB', 'GT', 'IT', 'KR', 'MX', 'PT', 'TR', 'UA', 'US', 'UY'
])
K_null = st.checkbox("Is K Null?", value=False)
K = None if K_null else st.number_input("K", value=0.0)
L = st.number_input("L")
M = st.number_input("M")
N = st.number_input("N")
O = st.number_input("O")
P = st.number_input("P")
Q = st.number_input("Q")
R = st.number_input("R")
S = st.number_input("S")
Monto = st.number_input("Monto")


# Button to Predict
if st.button("Predict"):
    # Create raw input dictionary
    raw_input = {
        'B': B, 'S': S, 'P': P, 'C': C, 'M': M, 'Q': Q,
        'N': N, 'H': H, 'D': D, 'A': A, 'R': R, 'O': O,
        'Monto': Monto, 'J': J, 'K': K
    }

    try:
        # Preprocess Input
        processed_data = preprocess_input(raw_input)

        # Make Prediction
        prediction = model.predict(processed_data)
        prediction_proba = model.predict_proba(processed_data)[:, 1]

        # Display Results
        st.subheader("Prediction")
        st.write("Fraudulent" if prediction[0] == 1 else "Not Fraudulent")
        st.write(f"Probability of Fraud: {prediction_proba[0]:.2f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

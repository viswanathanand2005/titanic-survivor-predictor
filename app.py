import streamlit as st
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('titanic_model.pkl', 'rb'))

st.title("🚢 Titanic Survivor Predictor")

# --- User Inputs ---
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.radio("Sex", ['male', 'female'])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", 0, 8, 0)
parch = st.number_input("Number of Parents/Children Aboard (Parch)", 0, 6, 0)
fare = st.number_input("Fare Paid ($)", 0.0, 600.0, 32.0)
embarked = st.selectbox("Port of Embarkation", ['S', 'C', 'Q'])

# --- Feature Engineering (just like src.py) ---
family_size = sibsp + parch

# Label encode sex manually (since you did it before OneHot in src.py)
sex_encoded = 1 if sex == 'male' else 0

# One-hot encode 'Embarked' (order: C, Q, S based on training)
embarked_c = 1 if embarked == 'C' else 0
embarked_q = 1 if embarked == 'Q' else 0
embarked_s = 1 if embarked == 'S' else 0

# Construct input vector in the same column order as in training
input_data = np.array([[embarked_c, embarked_q, embarked_s,
                        pclass, sex_encoded, age, sibsp, parch, fare, family_size]])

# Predict
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("🎉 The passenger would have **SURVIVED**.")
    else:
        st.error("☠️ The passenger would have **NOT SURVIVED**.")

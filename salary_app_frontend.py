import numpy as np
import streamlit as st
import pickle

model = pickle.load(open("linear_regression_model.pkl", "rb"))

st.title('Salary prediction app')

st.write('This app predicts the salary with respect to the years of experience')

years_experience = st.number_input("Enter years of experience :", min_value=0.0, max_value=50.0, value = 1.0, step=0.5)

if st.button("Predict salary"):
    experience_input = np.array([[years_experience]])
    prediction = model.predict(experience_input)
    
    st.success(f'The predicted salary is {years_experience} years of experience is : {prediction[0]:,.2f} INR')
    
st.write("This prediction is formed from the salary and years of experience dataset")

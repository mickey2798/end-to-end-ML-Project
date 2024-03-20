import streamlit as st
import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

## Route for a home page

def index():
    st.title('Home Page')
    st.write("Welcome to the home page. Fill in the form below to predict data point.")
    st.write("---")

def predict_datapoint():
    st.title('Predict Data Point')
    st.write("Fill in the details below to predict.")
    st.write("---")
    gender = st.selectbox("Gender", ["male", "female"])
    ethnicity = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
    parental_level_of_education = st.selectbox("Parental Level of Education", ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"])
    lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
    test_preparation_course = st.selectbox("Test Preparation Course", ["none", "completed"])
    reading_score = st.number_input("Reading Score")
    writing_score = st.number_input("Writing Score")

    if st.button("Predict"):
        data = CustomData(
            gender=gender,
            race_ethnicity=ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score
        )
        pred_df = data.get_data_as_data_frame()
        st.write(pred_df)
        st.write("Before Prediction")

        predict_pipeline = PredictPipeline()
        st.write("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        st.write("After Prediction")
        st.write("Prediction Result:", results[0])

def main():
    index()
    predict_datapoint()

if __name__ == "__main__":
    main()

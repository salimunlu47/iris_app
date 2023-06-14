#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import seaborn as sns
import joblib

# Load the dataset
iris_data = sns.load_dataset('iris')

# Load the trained models
trained_models = joblib.load('trained_models.joblib')

# Load the label encoder
label_encoder = joblib.load('label_encoder.joblib')

# Load the performance metrics
performance_metrics = joblib.load('performance_metrics.joblib')

# Title and description
st.title("Iris Classification App")
st.markdown("This app predicts the species of iris flowers based on the input measurements.")
st.markdown("Please adjust the sliders to input the iris measurements.")

# Checkbox to show the training dataframe
if st.checkbox("Show Training DataFrame"):
    st.write(iris_data)  # Display the training dataframe

# Display the iris image
image = 'iris_image.png'
st.image(image, use_column_width=True)

# Sidebar inputs
st.sidebar.title("Enter Iris Measurements")
sepal_length = st.sidebar.slider("Sepal Length", float(iris_data["sepal_length"].min()), float(iris_data["sepal_length"].max()))
sepal_width = st.sidebar.slider("Sepal Width", float(iris_data["sepal_width"].min()), float(iris_data["sepal_width"].max()))
petal_length = st.sidebar.slider("Petal Length", float(iris_data["petal_length"].min()), float(iris_data["petal_length"].max()))
petal_width = st.sidebar.slider("Petal Width", float(iris_data["petal_width"].min()), float(iris_data["petal_width"].max()))

# Model selection
model_names = list(trained_models.keys())
selected_model = st.sidebar.selectbox("Select Model", model_names)

# Get the selected model
model = trained_models[selected_model]

# Make a prediction
features = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(features)
predicted_species = label_encoder.inverse_transform(prediction)[0]

# Display the prediction in a box with colorful, big font
# Display the predicted species in blue
st.subheader("Prediction:")
prediction_text = f"<span style='color: blue; font-size: 24px;'>{predicted_species}</span>"
st.markdown(prediction_text, unsafe_allow_html=True)

# Update the performance metrics based on the selected model
model_metrics = performance_metrics[selected_model]

# Display the performance metrics
st.subheader("Performance Metrics:")
st.write("Accuracy:", model_metrics['accuracy'])
st.write("Precision:", model_metrics['precision'])
st.write("Recall:", model_metrics['recall'])
st.write("F1 Score:", model_metrics['f1_score'])

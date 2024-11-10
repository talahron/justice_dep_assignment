Lawsuit Prediction Model Application

Overview

This repository includes two parts of the same project: a machine learning model for predicting lawsuit outcomes and an application for making predictions using a user-friendly interface.

Part 1: Model and Training

Data Preprocessing and Model Training: The data_preprocessing_pipeline.py script preprocesses raw data, performs feature engineering, and trains multiple classification models (Logistic Regression, Decision Tree, Random Forest, XGBoost). The best-performing model is saved as model.pkl.



Part 2: Gradio Application

Application: The app.py script provides a Gradio-based web interface that loads the trained model for predicting lawsuit outcomes.


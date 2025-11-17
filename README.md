# Wine-Quality-ML-API

This repository contains two main branches that serve different aspects of the project. One branch is focused on the machine learning model for wine quality prediction, and the other is focused on deploying the model as a Flask API. 

## Branches

1. **Wine-ML-Model**: 
   - This branch contains the code and models for the machine learning part of the project, which predicts wine quality based on various features of the wine. 
   - The machine learning models used include AdaBoost, Gradient Boosting, and XGBoost.
   - The trained model is saved in a `.pkl` file for future predictions.

2. **Flask-API**:
   - This branch contains the Flask API that serves the machine learning model. The API allows users to send HTTP requests with wine feature data and receive the predicted quality as a response.
   - The `modelEn.pkl` file is used to load the pre-trained model and make predictions.


## Wine Quality Prediction (ML)

The main goal of this project is to predict the quality of red wine based on 11 features such as acidity, sugar, pH, alcohol, etc.

### Process

1. **Data Preprocessing**: 
   - Loaded and cleaned the dataset.
   - Created a new binary target column to classify wines as 'Good' or 'Bad' based on their quality (greater than or equal to 7).

2. **Exploratory Data Analysis (EDA)**: 
   - Visualized correlations and relationships between features.
   - Used heatmaps to explore feature importance.

3. **Model Training**:
   - Trained three different machine learning models: AdaBoost, Gradient Boosting, and XGBoost.
   - Evaluated the models based on accuracy and other classification metrics.

4. **Model Export**: 
   - The best-performing model (XGBoost) was saved using `pickle` for future use in the Flask API.

### To run the ML model:

1. Run the script:
   ```bash
   python REDWINE_Ensemble.py
   ```

## Flask API

This section contains the Flask API that serves the wine quality prediction model. The API allows users to submit wine feature data and receive predictions about its quality.

### Setup

1. Run the Flask app:
   ```bash
   python app.py
   ```

The Flask app will run locally, and you can access the API at `http://127.0.0.1:5000`.

### API Endpoints

1. **POST `/predict`**: 
   - This endpoint accepts wine features as input in JSON format and returns the predicted quality of the wine (0 for bad, 1 for good).
   - Example Request:
     ```json
     {
       "fixed_acidity": 7.4,
       "volatile_acidity": 0.7,
       "citric_acid": 0.0,
       "residual_sugar": 1.9,
       "chlorides": 0.076,
       "free_sulfur_dioxide": 11.0,
       "total_sulfur_dioxide": 34.0,
       "density": 0.9978,
       "pH": 3.51,
       "sulphates": 0.56,
       "alcohol": 9.4
     }
     ```

   - Example Response:
     ```json
     {
       "prediction": 1
     }
     ```

## Requirements

- Python 3.6 or higher
- Flask
- scikit-learn
- XGBoost
- pandas
- numpy
- seaborn
- matplotlib
- pickle




# 2025aa05903_ml_assignment2

## a) Problem Statement

- Implement multiple classification models 
- Build an interactive Streamlit web application to demonstrate our models 
- Deploy the app on Streamlit Community Cloud (FREE) 
- Share clickable links for evaluation 

## b) Dataset description

- This dataset includes two types of wines white and red.
- This has many i/p features density, pH, sulphates, chlorides, alochol, color.
- Target was quality of the wine.
- This is a classification task.

## c) Models used

- Random Forest
- XGBoost              
- Decision Tree
- K-Nearest
- Logistic Regression
- Naive Bayes

### Model Performance Comparison

-------------------------------------------------------------------------------------------
| Model                | Accuracy | AUC Score | Precision | Recall   | F1-Score | MCC     |
|----------------------|----------|-----------|-----------|----------|----------|---------|
| Random Forest        | 0.669231 | 0.850631  | 0.673772  | 0.669231 | 0.657380 | 0.482480 |
| XGBoost              | 0.655385 | 0.825920  | 0.646191  | 0.655385 | 0.646375 | 0.466446 |
| Decision Tree        | 0.594615 | 0.699867  | 0.598428  | 0.594615 | 0.594627 | 0.399938 |
| K-Nearest Neighbors  | 0.545385 | 0.733073  | 0.532721  | 0.545385 | 0.534968 | 0.299032 |
| Logistic Regression  | 0.536154 | 0.711083  | 0.500927  | 0.536154 | 0.501930 | 0.250956 |
| Naive Bayes          | 0.465385 | 0.670613  | 0.488536  | 0.465385 | 0.460981 | 0.237242 |
--------------------------------------------------------------------------------------------

### Model-wise Performance Observations

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| **Logistic Regression** | Achieved an accuracy of **0.536** and MCC of **0.251**, indicating limited performance. As a linear model, it struggled to capture the complex, non-linear relationships present in the dataset. |
| **Decision Tree** | With an accuracy of **0.595** and MCC of **0.400**, the Decision Tree performed better than linear models but showed instability due to overfitting, which limits its generalization ability. |
| **kNN** | The kNN model obtained an accuracy of **0.545** and MCC of **0.299**. Its distance-based nature made it sensitive to feature dimensionality, resulting in only moderate predictive performance. |
| **Naive Bayes** | Recorded the lowest accuracy of **0.465** and MCC of **0.237**. The strong independence assumption among features likely reduced its effectiveness for this dataset. |
| **Random Forest (Ensemble)** | This was the best-performing model with an accuracy of **0.669** and the highest MCC of **0.482**. The ensemble of multiple trees effectively reduced variance and captured complex feature interactions. |
| **XGBoost (Ensemble)** | Achieved strong performance with an accuracy of **0.655** and MCC of **0.466**. Its gradient boosting framework handled non-linear patterns well, making it a close competitor to Random Forest. |


Link to the Streamlit app: [Website link](https://2025aa05903mlassignment2.streamlit.app/)

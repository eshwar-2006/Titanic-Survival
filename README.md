# Titanic-Survival
# Titanic Survival Prediction Project

## 📊 Project Introduction

This project tackles the classic Kaggle "Titanic: Machine from Disaster" competition. The goal is to build a machine learning model that accurately predicts whether a passenger survived the sinking of the Titanic. For anyone venturing into data science, this project serves as an excellent introduction to a complete machine learning workflow, from initial data exploration and cleaning to advanced feature engineering and predictive modeling. This notebook demonstrates how to transform a raw, historical dataset into a powerful predictive tool, enabling us to uncover the key factors that determined survival for those aboard the ill-fated voyage.

## ✨ Key Features & Technical Implementation

This analysis was built using Python and popular data science libraries, incorporating a comprehensive machine learning pipeline.

* **⚙️ Data Exploration & Cleaning:** Used **Pandas** and **Seaborn** to investigate the dataset, identify missing values, and understand initial relationships between variables. Null values in 'Age', 'Fare', and 'Embarked' were strategically imputed.

* **🔧 Advanced Feature Engineering:** Created new, insightful features to improve model performance:
    * **Family Size:** Combined 'SibSp' and 'Parch' to create a 'Family_Size' feature, which was then grouped into categories like 'Alone' and 'Small'.
    * **Title Extraction:** Extracted passenger titles (e.g., 'Mr', 'Mrs', 'Miss') from the 'Name' column to create a 'Title' feature, which was later consolidated into broader social categories.
    * **Numerical Binning:** Transformed continuous 'Age' and 'Fare' data into ordinal categories to better capture their impact on survival.
    * **Cabin Information:** Created a binary feature 'Cabin\_Assign' to distinguish between passengers with and without an assigned cabin.

* **🤖 Model Building with Scikit-learn:**
    * **Preprocessing Pipeline:** Developed a robust preprocessing pipeline using `ColumnTransformer` to handle both categorical and numerical data. This included `OneHotEncoder` for nominal features ('Sex', 'Embarked') and `OrdinalEncoder` for created features like 'Family\_size\_Grouped'.
    * **Hyperparameter Tuning:** Employed `GridSearchCV` with `StratifiedKFold` cross-validation to systematically find the optimal hyperparameters for multiple classification algorithms.
    * **Model Training:** Trained and evaluated a diverse set of classifiers:
        * Logistic Regression
        * K-Nearest Neighbors (KNN)
        * Support Vector Machine (SVC)
        * Decision Tree Classifier
        * Random Forest Classifier
        * Gaussian Naive Bayes
        * AdaBoost Classifier
        * Gradient Boosting Classifier

* **📈 Model Evaluation & Prediction:**
    * Each model's performance was assessed based on its cross-validation score on the training data.
    * The tuned models were then used to predict survival on the unseen test dataset.
    * The final predictions for each model were exported to separate CSV files for submission.

## Conclusion

This project successfully demonstrates a complete, end-to-end machine learning workflow. By methodically cleaning the data, engineering insightful features, and systematically evaluating multiple models, we transformed the raw Titanic dataset into a robust predictive solution. The use of `scikit-learn` pipelines ensured that the entire process was reproducible and efficient. The final result is a set of optimized models capable of predicting passenger survival with a high degree of accuracy, showcasing the power of data science to extract actionable insights from historical data.

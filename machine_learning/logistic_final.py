import pandas as pd
import numpy as np
from io import StringIO
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Step 1: Download and Save CSV
file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
response = requests.get(file_path)

if response.status_code == 200:
    with open("ChurnData.csv", "w", encoding="utf-8") as f:
        f.write(response.text)
    print("Data successfully saved to ChurnData.csv")

    # Load into pandas
    data = StringIO(response.text)
    df = pd.read_csv(data)

    # --- Data Preprocessing ---
    # Select relevant features
    churn_df = df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'churn']]
    X = churn_df.drop('churn', axis=1).values
    y = churn_df['churn'].values

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    # --- Model Training ---
    LR = LogisticRegression()
    LR.fit(X_train, y_train)

    # --- Predictions ---
    yhat = LR.predict(X_test)
    print("\nSample Predictions:")
    print(yhat[:10])

    # --- Prediction Probabilities ---
    yhat_prob = LR.predict_proba(X_test)
    print("\nPrediction Probabilities (first 10 rows):")
    print(yhat_prob[:10])

    # --- Feature Importance ---
    coefficients = pd.Series(LR.coef_[0], index=churn_df.columns[:-1])
    coefficients.sort_values().plot(kind='barh', figsize=(8,5))
    plt.title("Feature Coefficients in Logistic Regression Churn Model")
    plt.xlabel("Coefficient Value")
    plt.tight_layout()
    plt.show()

    # --- Model Evaluation ---
    loss = log_loss(y_test, yhat_prob)
    print(f"\nLog Loss of Model: {loss:.4f}")
    print("\nInterpretation:")
    print("- Lower log loss indicates a better calibrated model.")
    print("- Large positive coefficients → higher chance of churn (class 1).")
    print("- Large negative coefficients → lower chance of churn.")
    print("- Near-zero coefficients → minimal influence on prediction.")

else:
    print(f"Failed to download file. Status code: {response.status_code}")


#multi linear regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
import requests
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Step 1: Download and Save CSV
file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
response = requests.get(file_path)

if response.status_code == 200:
    with open("FuelConsumption.csv", "w", encoding="utf-8") as f:
        f.write(response.text)
    print("Data successfully saved to FuelConsumption.csv")

    # Load into pandas
    data = StringIO(response.text)
    df = pd.read_csv(data)

    # Keep only required columns
    df = df[['ENGINESIZE', 'FUELCONSUMPTION_COMB_MPG', 'CO2EMISSIONS']].dropna()

    # --- 1. LINEAR REGRESSION on ENGINESIZE ---
    print("\n--- Regression Using ENGINESIZE ---")
    X = df[['ENGINESIZE']].values
    y = df[['CO2EMISSIONS']].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    reg_eng = LinearRegression()
    reg_eng.fit(X_train, y_train)

    print(f"Coefficient: {reg_eng.coef_[0][0]:.2f}")
    print(f"Intercept: {reg_eng.intercept_[0]:.2f}")

    # Training plot
    plt.scatter(X_train, y_train, color='blue', label='Training Data')
    plt.plot(X_train, reg_eng.predict(X_train), color='red', label='Best Fit Line')
    plt.xlabel("Engine size")
    plt.ylabel("CO2 Emissions")
    plt.title("Training: ENGINESIZE vs CO2 Emissions")
    plt.legend()
    plt.show()

    # Test plot
    plt.scatter(X_test, y_test, color='green', label='Test Data')
    plt.plot(X_test, reg_eng.predict(X_test), color='red', label='Best Fit Line')
    plt.xlabel("Engine size")
    plt.ylabel("CO2 Emissions")
    plt.title("Test: ENGINESIZE vs CO2 Emissions")
    plt.legend()
    plt.show()

    # --- 2. LINEAR REGRESSION on FUELCONSUMPTION_COMB_MPG ---
    print("\n--- Regression Using FUELCONSUMPTION_COMB_MPG ---")
    X_mpg = df[['FUELCONSUMPTION_COMB_MPG']].values
    y_mpg = df[['CO2EMISSIONS']].values

    X_train_mpg, X_test_mpg, y_train_mpg, y_test_mpg = train_test_split(X_mpg, y_mpg, test_size=0.2, random_state=1)

    reg_mpg = LinearRegression()
    reg_mpg.fit(X_train_mpg, y_train_mpg)

    print(f"Coefficient: {reg_mpg.coef_[0][0]:.2f}")
    print(f"Intercept: {reg_mpg.intercept_[0]:.2f}")

    # Test plot for MPG
    plt.scatter(X_test_mpg, y_test_mpg, color='purple', label='Test Data')
    plt.plot(X_test_mpg, reg_mpg.predict(X_test_mpg), color='orange', label='Best Fit Line')
    plt.xlabel("FUELCONSUMPTION_COMB_MPG")
    plt.ylabel("CO2 Emissions")
    plt.title("Test: Fuel Consumption (MPG) vs CO2 Emissions")
    plt.legend()
    plt.show()

    print("\n--- Model Evaluation Thoughts ---")
    print("The fit for both models is visually reasonable, but likely insufficient for capturing all variability.")
    print("- Engine size shows a positive linear trend, but with some spread.")
    print("- Fuel consumption (MPG) has an inverse relationship with emissions, and may benefit from polynomial features.")
    print("To improve model accuracy, consider including additional features or trying polynomial regression.")

else:
    print(f"Failed to download file. Status code: {response.status_code}")


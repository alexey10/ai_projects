import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
import requests
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
    print(df.describe())

    # Select key features
    cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
    print("\nSample of selected features:\n", cdf.sample(9))

    # Visualize histograms
    cdf[['CYLINDERS','ENGINESIZE','FUELCONSUMPTION_COMB','CO2EMISSIONS']].hist(figsize=(10,8))
    plt.suptitle("Histograms of Key Features")
    plt.tight_layout()
    plt.show()

    # Scatter plots
    plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
    plt.xlabel("FUELCONSUMPTION_COMB")
    plt.ylabel("CO2 Emissions")
    plt.title("Fuel Consumption vs CO2 Emissions")
    plt.show()

    plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
    plt.xlabel("Engine Size")
    plt.ylabel("CO2 Emissions")
    plt.title("Engine Size vs CO2 Emissions")
    plt.xlim(0, 7)
    plt.show()

    # Step 2: Extract input and output
    X = cdf[['ENGINESIZE']].values
    y = cdf['CO2EMISSIONS'].values

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Step 3: Fit Linear Regression model
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Plot the fit line
    plt.scatter(X_train, y_train, color='blue')
    plt.plot(X_train, regressor.predict(X_train), '-r')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    plt.title("Regression Line on Training Data")
    plt.show()

    # Step 4: Evaluate model on test data
    y_test_pred = regressor.predict(X_test)

    print("\nModel Evaluation Metrics:")
    print("Mean Absolute Error: %.2f" % mean_absolute_error(y_test, y_test_pred))
    print("Mean Squared Error: %.2f" % mean_squared_error(y_test, y_test_pred))
    print("Root Mean Squared Error: %.2f" % np.sqrt(mean_squared_error(y_test, y_test_pred)))
    print("R2 Score: %.2f" % r2_score(y_test, y_test_pred))

else:
    print(f"Failed to download file. Status code: {response.status_code}")


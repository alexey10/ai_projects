import pandas as pd
import requests
from io import StringIO
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# URL of the dataset
file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv"

# Download the CSV content
response = requests.get(file_path)

if response.status_code == 200:
    # Save locally
    with open("houses_data.csv", "w", encoding="utf-8") as f:
        f.write(response.text)

    # Load into DataFrame
    df = pd.read_csv(StringIO(response.text))

    # Drop irrelevant columns
    df.drop(["id", "Unnamed: 0"], axis=1, inplace=True)

    # Drop any remaining rows with NaN values
    df.dropna(inplace=True)


    # Replace missing values
    df["bedrooms"].replace(to_replace=pd.NA, value=df["bedrooms"].mean(), inplace=True)
    df["bathrooms"].replace(to_replace=pd.NA, value=df["bathrooms"].mean(), inplace=True)

    # --------- VALUE COUNTS AND VISUALS ---------
    print("\nHouse counts by floor value:")
    print(df['floors'].value_counts().to_frame())

    # Boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='waterfront', y='price', data=df)
    plt.title("Price Distribution by Waterfront")
    plt.show()

    # Regplot
    plt.figure(figsize=(8, 6))
    sns.regplot(x='sqft_above', y='price', data=df)
    plt.title("Price vs. Sqft Above")
    plt.show()

    # Correlation
    corr_matrix = df.corr(numeric_only=True)
    price_corr = corr_matrix['price'].drop('price')
    top_feature = price_corr.abs().idxmax()
    print(f"\nMost correlated feature with price: {top_feature} ({price_corr[top_feature]:.4f})")

    
    # --------- REMOVE REMAINING NaN VALUES ---------
    df.dropna(inplace=True)


    # --------- LINEAR REGRESSION MODELS ---------
    lr = LinearRegression()

    # 1. Predict price using 'long'
    x_long = df[["long"]]
    y = df["price"]
    lr.fit(x_long, y)
    r2_long = lr.score(x_long, y)
    print(f"\nR² using 'long' as predictor: {r2_long:.4f}")

    # 2. Predict price using 'sqft_living'
    x_sqft = df[["sqft_living"]]
    lr.fit(x_sqft, y)
    r2_sqft = lr.score(x_sqft, y)
    print(f"R² using 'sqft_living' as predictor: {r2_sqft:.4f}")

    # 3. Predict price using multiple features
    features = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement", 
                "view", "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"]
    X_multi = df[features]
    lr.fit(X_multi, y)
    r2_multi = lr.score(X_multi, y)
    print(f"R² using multiple features: {r2_multi:.4f}")

else:
    print(f"Failed to download file. Status code: {response.status_code}")


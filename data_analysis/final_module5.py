import pandas as pd
import requests
from io import StringIO
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Step 1: Download and Save CSV
file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv"
response = requests.get(file_path)

if response.status_code == 200:
    with open("houses_data.csv", "w", encoding="utf-8") as f:
        f.write(response.text)
    print("Data successfully saved to houses_data.csv")

    data = StringIO(response.text)
    df = pd.read_csv(data)

    # Step 2: Clean Missing Values
    df["bedrooms"].replace(to_replace=pd.NA, value=df["bedrooms"].mean(), inplace=True)
    df["bathrooms"].replace(to_replace=pd.NA, value=df["bathrooms"].mean(), inplace=True)
    df.dropna(inplace=True)

    # Step 3: Value counts of floors
    floors_count = df['floors'].value_counts().to_frame()
    print("\nFloor value counts:\n", floors_count)

    # Step 4: Boxplot for waterfront vs price
    sns.boxplot(x="waterfront", y="price", data=df)
    plt.title("Waterfront vs Price (Outlier Detection)")
    plt.show()

    # Step 5: Regplot for sqft_above vs price
    sns.regplot(x="sqft_above", y="price", data=df)
    plt.title("Correlation between sqft_above and price")
    plt.show()

    # Step 6: Correlation with price
    correlation = df.corr(numeric_only=True)["price"].drop("price").sort_values(ascending=False)
    print("\nTop correlated feature with price:\n", correlation.head(1))

    # Step 7: Linear model using 'long'
    lm = LinearRegression()
    X_long = df[["long"]]
    y = df["price"]
    lm.fit(X_long, y)
    print("\nR^2 using 'long':", lm.score(X_long, y))

    # Step 8: Linear model using 'sqft_living'
    X_sqft = df[["sqft_living"]]
    lm.fit(X_sqft, y)
    print("R^2 using 'sqft_living':", lm.score(X_sqft, y))

    # Step 9: Linear model using multiple features
    features = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement",
                "view", "bathrooms", "sqft_living15", "sqft_above",
                "grade", "sqft_living"]
    X = df[features]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    lm.fit(X_train, y_train)
    print("R^2 using multiple features:", lm.score(X_test, y_test))

    # Step 10: Polynomial Pipeline
    Input = [
        ('scale', StandardScaler()),
        ('polynomial', PolynomialFeatures(include_bias=False)),
        ('model', LinearRegression())
    ]
    pipe = Pipeline(Input)
    pipe.fit(X_train, y_train)
    print("R^2 using Polynomial Regression Pipeline:", pipe.score(X_test, y_test))

    # Step 11: Train-Test Split for Ridge
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)
    print("\nnumber of training samples:", x_train.shape[0])
    print("number of test samples:", x_test.shape[0])

    # Step 12: Ridge Regression
    ridge_model = Ridge(alpha=0.1)
    ridge_model.fit(x_train, y_train)
    print("R^2 on test data using Ridge Regression:", ridge_model.score(x_test, y_test))

    # Step 13: Polynomial Ridge Regression
    poly = PolynomialFeatures(degree=2, include_bias=False)
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.transform(x_test)

    ridge_poly = Ridge(alpha=0.1)
    ridge_poly.fit(x_train_poly, y_train)
    print("R^2 on test data using Polynomial Ridge Regression:", ridge_poly.score(x_test_poly, y_test))

else:
    print(f"Failed to download file. Status code: {response.status_code}")


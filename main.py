"""CodeAlpha Task 4: Sales Prediction using Python

Expected CSV columns: TV, Radio, Newspaper, Sales
Download the Advertising.csv dataset linked in the CodeAlpha task PDF.
Run: python main.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_FILE = Path("data/Advertising.csv")
OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)

# 1. Read and inspect the data
if not DATA_FILE.exists():
    raise FileNotFoundError(
        "Dataset not found. Download Advertising.csv from "
        "https://www.kaggle.com/datasets/bumba5341/advertisingcsv "
        "and place it inside the data folder as data/Advertising.csv"
    )

df = pd.read_csv(DATA_FILE)
# Some versions of the dataset include an extra unnamed index column.
df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False)]

required = {"TV", "Radio", "Newspaper", "Sales"}
missing_columns = required - set(df.columns)
if missing_columns:
    raise ValueError(f"CSV is missing required columns: {sorted(missing_columns)}")

print("\nFirst five rows:\n", df.head())
print("\nDataset information:")
df.info()
print("\nMissing values:\n", df.isnull().sum())

# 2. Clean rows with missing values in the required columns
df = df.dropna(subset=list(required)).copy()
features = ["TV", "Radio", "Newspaper"]
X = df[features]
y = df["Sales"]

# 3. Explore relationships between advertising and sales
for feature in features:
    plt.figure()
    plt.scatter(df[feature], y, alpha=0.7)
    plt.xlabel(f"{feature} advertising spend")
    plt.ylabel("Sales")
    plt.title(f"{feature} Advertising vs Sales")
    plt.tight_layout()
    plt.savefig(OUTPUT / f"{feature.lower()}_vs_sales.png", dpi=150)
    plt.close()

# 4. Split data and train a multiple linear regression model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict and evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

print("\nModel evaluation")
print(f"MAE : {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R-squared: {r2:.3f}")
print("\nModel coefficients:")
for name, value in zip(features, model.coef_):
    print(f"{name}: {value:.3f}")
print(f"Intercept: {model.intercept_:.3f}")

# 6. Save actual vs predicted chart
plt.figure()
plt.scatter(y_test, predictions, alpha=0.7)
low = min(y_test.min(), predictions.min())
high = max(y_test.max(), predictions.max())
plt.plot([low, high], [low, high], linestyle="--")
plt.xlabel("Actual sales")
plt.ylabel("Predicted sales")
plt.title("Actual vs Predicted Sales")
plt.tight_layout()
plt.savefig(OUTPUT / "actual_vs_predicted.png", dpi=150)
plt.close()

# 7. Save predictions and metrics
pd.DataFrame({"actual_sales": y_test, "predicted_sales": predictions}).to_csv(
    OUTPUT / "predictions.csv", index=False
)
(OUTPUT / "evaluation.txt").write_text(
    f"Model: Multiple Linear Regression\nRows used: {len(df)}\n"
    f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR-squared: {r2:.4f}\n",
    encoding="utf-8"
)

# 8. Example prediction: replace these amounts with values in the dataset's units
example = pd.DataFrame([[150, 25, 20]], columns=features)
print("\nExample predicted sales for TV=150, Radio=25, Newspaper=20:",
      round(model.predict(example)[0], 2))
print("\nCharts, predictions and evaluation saved in outputs/.")

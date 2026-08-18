import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CAR PRICE PREDICTION SYSTEM
# STAGE 3 - MACHINE LEARNING MODEL
# ============================================================

print("=" * 65)
print("                 CAR PRICE PREDICTION SYSTEM")
print("=" * 65)


# ============================================================
# 1. LOAD CLEANED DATASET
# ============================================================

print("\nLoading cleaned dataset...")

data = pd.read_csv("cleaned_car_data.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)


# ============================================================
# 2. DISPLAY DATA
# ============================================================

print("\nFirst 5 records:")
print(data.head())


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 65)
print("FEATURE ENGINEERING")
print("=" * 65)

# Current reference year
CURRENT_YEAR = 2026

# Create car age
data["car_age"] = CURRENT_YEAR - data["year"]

print("\nCreated feature: car_age")

print(data[["year", "car_age"]].head())


# ============================================================
# 4. REMOVE UNNECESSARY COLUMN
# ============================================================

# Car name has many unique values and is not useful for
# our first model.

data = data.drop(columns=["name"])


# ============================================================
# 5. DEFINE FEATURES AND TARGET
# ============================================================

X = data.drop(columns=["Price"])

y = data["Price"]


print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print("Price")


# ============================================================
# 6. IDENTIFY COLUMN TYPES
# ============================================================

categorical_features = [
    "company",
    "fuel_type"
]

numerical_features = [
    "year",
    "kms_driven",
    "car_age"
]


print("\nCategorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)


# ============================================================
# 7. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\n" + "=" * 65)
print("TRAIN TEST SPLIT")
print("=" * 65)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 8. DATA PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ============================================================
# 9. RANDOM FOREST MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 10. CREATE ML PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ============================================================
# 11. TRAIN MODEL
# ============================================================

print("\n" + "=" * 65)
print("MODEL TRAINING")
print("=" * 65)

print("\nTraining Random Forest model...")

pipeline.fit(
    X_train,
    y_train
)

print("\nModel training completed successfully!")


# ============================================================
# 12. MAKE TEST PREDICTIONS
# ============================================================

y_pred = pipeline.predict(X_test)


# ============================================================
# 13. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n" + "=" * 65)
print("MODEL PERFORMANCE")
print("=" * 65)

print(f"\nMean Absolute Error : ₹{mae:,.2f}")
print(f"Root Mean Squared Error : ₹{rmse:,.2f}")
print(f"R² Score : {r2:.4f}")
print(f"R² Accuracy : {r2 * 100:.2f}%")


# ============================================================
# 14. SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 65)
print("SAMPLE PREDICTIONS")
print("=" * 65)

sample_results = pd.DataFrame({
    "Actual Price": y_test.values[:10],
    "Predicted Price": y_pred[:10]
})

sample_results["Difference"] = (
    sample_results["Actual Price"]
    - sample_results["Predicted Price"]
)

print("\n")

for index, row in sample_results.iterrows():

    print(
        f"Actual: ₹{row['Actual Price']:,.0f} | "
        f"Predicted: ₹{row['Predicted Price']:,.0f}"
    )


# ============================================================
# 15. SAVE TRAINED MODEL
# ============================================================

model_file = "car_price_model.pkl"

joblib.dump(
    pipeline,
    model_file
)

print("\n" + "=" * 65)
print("MODEL SAVED")
print("=" * 65)

print(f"\nTrained model saved as:")
print(model_file)


# ============================================================
# 16. TEST NEW CAR
# ============================================================

print("\n" + "=" * 65)
print("NEW CAR PRICE PREDICTION TEST")
print("=" * 65)


# Example car

new_car = pd.DataFrame({
    "company": ["Maruti"],
    "year": [2018],
    "kms_driven": [22000],
    "fuel_type": ["Petrol"],
    "car_age": [CURRENT_YEAR - 2018]
})


predicted_price = pipeline.predict(
    new_car
)[0]


print("\nSample Car Details:")
print("----------------------------")
print("Company     : Maruti")
print("Year        : 2018")
print("KMs Driven  : 22,000")
print("Fuel Type   : Petrol")
print("----------------------------")

print(
    f"\nPredicted Car Price: ₹{predicted_price:,.0f}"
)


# ============================================================
# 17. COMPLETION
# ============================================================

print("\n" + "=" * 65)
print("        MACHINE LEARNING TRAINING COMPLETED")
print("=" * 65)

print("\nNext step:")
print("Create GUI for entering car details")
print("and predicting the estimated price.")
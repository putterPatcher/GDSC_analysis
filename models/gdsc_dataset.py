import pandas as pd
import matplotlib.pyplot as plt

gdsc_data_1_path = "../Data/GDSC_DATASET.csv"

df = pd.read_csv(gdsc_data_1_path)

print(df.dtypes)

df1 = pd.DataFrame()

# Data screening

df1 = pd.DataFrame(df[["TCGA_DESC", "DRUG_NAME", "LN_IC50", "AUC", "Z_SCORE", "Microsatellite instability Status (MSI)"]])
df1.rename(columns={"Microsatellite instability Status (MSI)": "MSI"}, inplace=True)

# Data cleaning

print(df1.dtypes)
print(df1.isna().sum())
df1.dropna(inplace=True)
print(df1.isna().sum())
print(df1.shape)

plt.scatter(df1[['AUC']].to_numpy()[:, 0], df1[['LN_IC50']].to_numpy()[:, 0])
plt.xlabel('AUC')
plt.ylabel('LN_IC50')
plt.show()

plt.scatter(df1[['Z_SCORE']].to_numpy()[:, 0], df1[['LN_IC50']].to_numpy()[:, 0])
plt.xlabel('Z_SCORE')
plt.ylabel('LN_IC50')
plt.show()

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Define which columns are nominal and which are ordinal
nominal_features = ['TCGA_DESC', 'DRUG_NAME', 'MSI']

# Create the transformers
nominal_transformer = OneHotEncoder(handle_unknown='ignore')

# Use ColumnTransformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('nom', nominal_transformer, nominal_features)
    ],
    remainder='passthrough' # Keep other columns (e.g., the target variable if present)
)

x = df1[['TCGA_DESC', 'DRUG_NAME', 'AUC', 'Z_SCORE', 'MSI']]
y = df1['LN_IC50']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=10, random_state=42, verbose=2, n_jobs=-1))])

# Fit the pipeline
pipeline.fit(x_train, y_train)

from sklearn.metrics import mean_squared_error, r2_score

y_pred = pipeline.predict(x_test)
# 5. Evaluate the model (using Mean Squared Error and R^2 score)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

with open('gdsc.json', 'w') as file:
    import json
    file.write(json.dumps([y_test.values.tolist(), y_pred.tolist()]))

from matplotlib import pyplot as plt

plt.plot(y_test, y_pred, color='green', marker='^')
plt.ylabel("Predicted")
plt.xlabel("Actual")
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

gdsc_data_2_path = "../Data/GDSC2-dataset.csv"

df = pd.read_csv(gdsc_data_2_path)

print(df.dtypes)

df1 = pd.DataFrame()

# Data screening

df1 = pd.DataFrame(df[["TCGA_DESC", "DRUG_NAME", "PATHWAY_NAME", "PUTATIVE_TARGET", "LN_IC50", "AUC", "Z_SCORE"]])

# Data cleaning

print(df1.dtypes)
print(df1.isna().sum())
df1.dropna(inplace=True)
print(df1.isna().sum())
print(df1.shape)

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Define which columns are nominal and which are ordinal
nominal_features = ['TCGA_DESC', 'DRUG_NAME', 'PATHWAY_NAME', 'PUTATIVE_TARGET']

# Create the transformers
nominal_transformer = OneHotEncoder(handle_unknown='ignore')

# Use ColumnTransformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('nom', nominal_transformer, nominal_features)
    ],
    remainder='passthrough' # Keep other columns (e.g., the target variable if present)
)

x = df1[['TCGA_DESC', 'DRUG_NAME', 'AUC', 'Z_SCORE', 'PATHWAY_NAME', 'PUTATIVE_TARGET']]
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

from matplotlib import pyplot as plt

plt.scatter([i for i in range(1,len(y_test)+1)], y_test, label='Actual')
plt.scatter([i for i in range(1,len(y_test)+1)], y_pred, color='red', marker='^', label='Predicted')
plt.ylabel("LN_IC50")
plt.legend()
plt.show()


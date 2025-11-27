import pandas as pd
import matplotlib.pyplot as plt

gdsc_data_1_path = "Data/GDSC_DATASET.csv"

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

# One way ANOVA

import pingouin as pg
print("\nLN_IC50")
print()
result = pg.anova(dv='LN_IC50', between=['TCGA_DESC'], data=df1)
print(result)
print()
result = pg.anova(dv='LN_IC50', between=['DRUG_NAME'], data=df1)
print(result)
print()
result = pg.anova(dv='LN_IC50', between=['MSI'], data=df1)
print(result)
print()
print("AUC\n")
result = pg.anova(dv='AUC', between=['TCGA_DESC'], data=df1)
print(result)
print()
result = pg.anova(dv='AUC', between=['DRUG_NAME'], data=df1)
print(result)
print()
result = pg.anova(dv='AUC', between=['MSI'], data=df1)
print(result)
print()
print("Z_Score\n")
result = pg.anova(dv='Z_SCORE', between=['TCGA_DESC'], data=df1)
print(result)
print()
result = pg.anova(dv='Z_SCORE', between=['DRUG_NAME'], data=df1)
print(result)
print()
result = pg.anova(dv='Z_SCORE', between=['MSI'], data=df1)
print(result)

plt.scatter(df1["LN_IC50"].to_numpy(), df1["AUC"].to_numpy())
plt.xlabel("LN_IC50")
plt.ylabel("AUC")
plt.show()

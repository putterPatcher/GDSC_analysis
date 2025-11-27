import pandas as pd
import matplotlib.pyplot as plt

gdsc_data_2_path = "Data/GDSC2-dataset.csv"

df = pd.read_csv(gdsc_data_2_path)

print(df.dtypes)

df1 = pd.DataFrame()

# Data screening

df1 = pd.DataFrame(df[["PATHWAY_NAME", "PUTATIVE_TARGET", "LN_IC50", "AUC", "Z_SCORE"]])

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
result = pg.anova(dv='LN_IC50', between=['PATHWAY_NAME'], data=df1)
print(result)
print()
result = pg.anova(dv='LN_IC50', between=['PUTATIVE_TARGET'], data=df1)
print(result)
print()
print("AUC\n")
result = pg.anova(dv='AUC', between=['PATHWAY_NAME'], data=df1)
print(result)
print()
result = pg.anova(dv='AUC', between=['PUTATIVE_TARGET'], data=df1)
print(result)
print()
print("Z_Score\n")
result = pg.anova(dv='Z_SCORE', between=['PATHWAY_NAME'], data=df1)
print(result)
print()
result = pg.anova(dv='Z_SCORE', between=['PUTATIVE_TARGET'], data=df1)
print(result)

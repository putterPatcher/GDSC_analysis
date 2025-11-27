import pandas as pd
import matplotlib.pyplot as plt

cell_line_details_path = "Data/Cell_Lines_Details.xlsx"

df = pd.read_excel(cell_line_details_path)

# Data Screening

df1 = pd.DataFrame()

df1["Cancer Type (matching TCGA label)"] = df["Cancer Type\n(matching TCGA label)"]
df1["Microsatellite instability Status (MSI)"] = df["Microsatellite \ninstability Status (MSI)"]
df1["Screen Medium"] = df["Screen Medium"]
df1["Growth Properties"] = df["Growth Properties"]

# Data cleaning

print(df1.isna().sum())
print(df1.shape)
df1.dropna(inplace=True)
print(df1.isna().sum())
print(df1.shape)

# Perform tests for independence (Chi-Square)

count_table = pd.crosstab(df1['Microsatellite instability Status (MSI)'], df1['Cancer Type (matching TCGA label)'])
print(count_table)
count_table1 = pd.crosstab(df1['Microsatellite instability Status (MSI)'], df1['Growth Properties'])
print(count_table1)
count_table2 = pd.crosstab(df1['Microsatellite instability Status (MSI)'], df1['Screen Medium'])
print(count_table2)

import scipy.stats as stats

print()
chi2 = stats.chi2_contingency(count_table)
print(chi2)
print()
chi2_1 = stats.chi2_contingency(count_table1)
print(chi2_1)
print()
chi2_2 = stats.chi2_contingency(count_table2)
print(chi2_2)

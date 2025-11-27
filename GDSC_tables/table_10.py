import seaborn as sns

from path import relative_dir, table_10
import pandas as pd
import matplotlib.pyplot as plt

path = relative_dir + table_10

df = pd.read_csv(path)

print(df.shape)

df1 = df.melt(['Drug'], value_name='PCorr', var_name='Drug1')

df2 = df1[((df1["PCorr"] > 0.5) | (df1["PCorr"] < -0.5)) & (df1["Drug"] != df1["Drug1"])]

print(df2)
print("Total:", df1.shape)
print("|r| > 0.5:", df2.shape)

df.set_index(df.columns[0], inplace=True)
sns.heatmap(df, cmap='gray')
plt.show()

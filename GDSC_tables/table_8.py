from path import relative_dir, table_8

path = relative_dir+table_8

import pandas as pd

df = pd.read_csv(path)

count_table = pd.crosstab(df['GeneA'], df['GeneB'])
print(count_table)

import scipy.stats as stats

print()
chi2 = stats.chi2_contingency(count_table)
print(chi2)

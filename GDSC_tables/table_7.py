from path import relative_dir, table_7

path = relative_dir+table_7

import pandas as pd
from randomization_test_for_interaction import interaction_permutation_gpu_batched
df = pd.read_csv(path)

df = df.melt(['COSMIC_ID'], var_name='Drug', value_name='AUC')

print(df.dtypes)
print(df.isna().sum())
df.dropna(inplace=True)
print(df.isna().sum())
print(df.shape)

result = interaction_permutation_gpu_batched(df, "COSMIC_ID", 'Drug', 'AUC', 100000)
print(result)
from path import relative_dir, table_2

path = relative_dir+table_2

import pandas as pd
from randomization_test_for_interaction import interaction_permutation_gpu_batched
df = pd.read_csv(path)

df = df.melt(['COSMIC_ID'], var_name='CPG_LOCI', value_name='METHYLATION')

print(df.dtypes)
print(df.isna().sum())
df.dropna(inplace=True)
print(df.isna().sum())
print(df.shape)

result = interaction_permutation_gpu_batched(df, "COSMIC_ID", 'CPG_LOCI', 'METHYLATION', )
print(result)

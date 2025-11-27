from path import relative_dir, table_9

path = relative_dir + table_9

import pandas as pd
from randomization_test_for_interaction import interaction_permutation_gpu_batched

df_dic = pd.read_excel(path, sheet_name=None)
for sheet_name, df in df_dic.items():
    if sheet_name == 'caption':
        continue
    df = df.melt(['Drug'], var_name='Feature', value_name='Presence')

    print(df.dtypes)
    print(df.isna().sum())
    df.dropna(inplace=True)
    print(df.isna().sum())
    print(df.shape)

    result = interaction_permutation_gpu_batched(df, "Drug", 'Feature', 'Presence', 100000)
    print(result)
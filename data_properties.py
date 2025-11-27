import pandas as pd
import os

cell_line_details_path = "Data/Cell_Lines_Details.xlsx"
compound_annotations_path = "Data/Compounds-annotation.csv"
gdsc_data_1_path = "Data/GDSC_DATASET.csv"
gdsc_data_2_path = "Data/GDSC2-dataset.csv"

def open_and_properties(path, type):
    if type == 'excel':
        df_dic = pd.read_excel(path, sheet_name=None)
        for sheet_name, df in df_dic.items():
            print(f"\n{sheet_name}:\n")
            print(df.head())
            print(df.tail())
            print(df.describe())
            print(df.dtypes)
            print(df.shape)
    elif type == 'csv':
        df = pd.read_csv(path)
        print(df.head())
        print(df.tail())
        print(df.describe())
        print(df.dtypes)
        print(df.shape)

print("Cell Line Details\n")
open_and_properties(cell_line_details_path, 'excel')
print("\n\nCompound Annotations\n")
open_and_properties(compound_annotations_path, 'csv')
print("\n\nGDSC_1\n")
open_and_properties(gdsc_data_1_path, 'csv')
print("\n\nGDSC_2\n")
open_and_properties(gdsc_data_2_path, 'csv')

os.chdir('./Data/GDSC_DATASET_S1-S12')
for i in os.listdir():
    j = i.split('.')
    print("\n\n{}\n".format(i))
    if j[1] == 'csv':
        open_and_properties(i, 'csv')
    else:
        open_and_properties(i, 'excel')

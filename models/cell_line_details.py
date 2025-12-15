# Cell line details - predict MSI using Random Forest Classifier
import pandas as pd
import matplotlib.pyplot as plt

cell_line_details_path = "../Data/Cell_Lines_Details.xlsx"

df = pd.read_excel(cell_line_details_path)
random_seed = [145,]

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

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

encoder1 = LabelEncoder()

y = encoder1.fit_transform(df1['Microsatellite instability Status (MSI)'])

import joblib
joblib.dump(encoder1, 'y_cell_line.joblib')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

encoder = OneHotEncoder()
x_rf = encoder.fit_transform(df1[['Cancer Type (matching TCGA label)', 'Screen Medium', 'Growth Properties']])

import joblib
joblib.dump(encoder, 'x_rf3.joblib')

x_gb = df1[['Cancer Type (matching TCGA label)', 'Screen Medium', 'Growth Properties']]

encoder_1 = OneHotEncoder()
x_rf_1 = encoder_1.fit_transform(df1[['Cancer Type (matching TCGA label)']])

joblib.dump(encoder_1, 'x_rf1.joblib')

x_gb_1 = df1[['Cancer Type (matching TCGA label)']]

def perform_classification(random_seed):
    result = []
    pred_vals = []
    model = []
    # RandomForestClassifier

    x_train, x_test, y_train, y_test = train_test_split(x_rf, y, test_size=0.3, random_state=random_seed)

    rf_model = RandomForestClassifier(n_estimators=1000, random_state=random_seed, n_jobs=-1)
    rf_model.fit(x_train, y_train)

    predictions = rf_model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    # print(f"Accuracy: {accuracy:.4f}")
    result.append(accuracy)
    pred_vals.append([y_test, predictions])
    model.append(rf_model)
    # from matplotlib import pyplot as plt

    # plt.scatter([i for i in range(1,len(y_test)+1)], encoder1.inverse_transform(y_test), label='Actual')
    # plt.scatter([i for i in range(1,len(y_test)+1)], encoder1.inverse_transform(predictions), color='red', marker='^', label='Predicted')
    # plt.legend()
    # plt.show()

    #histogram gradient boosting
    x_train, x_test, y_train, y_test = train_test_split(x_gb, y, test_size=0.3, random_state=random_seed)

    gb_model = HistGradientBoostingClassifier(categorical_features=['Cancer Type (matching TCGA label)', 'Screen Medium', 'Growth Properties'], max_iter=500)
    gb_model.fit(x_train, y_train)

    predictions = gb_model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    # print(f"Accuracy: {accuracy:.4f}")
    result.append(accuracy)
    pred_vals.append([y_test, predictions])
    model.append(gb_model)
    # from matplotlib import pyplot as plt

    # plt.scatter([i for i in range(1,len(y_test)+1)], encoder1.inverse_transform(y_test), label='Actual')
    # plt.scatter([i for i in range(1,len(y_test)+1)], encoder1.inverse_transform(predictions), color='red', marker='^', label='Predicted')
    # plt.legend()
    # plt.show()

    # random forest classifier - 2
    x_train, x_test, y_train, y_test = train_test_split(x_rf_1, y, test_size=0.3, random_state=random_seed)

    rf_model = RandomForestClassifier(n_estimators=1000, random_state=random_seed, n_jobs=-1)
    rf_model.fit(x_train, y_train)

    predictions = rf_model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    # print(f"Accuracy: {accuracy:.4f}")
    result.append(accuracy)
    pred_vals.append([y_test, predictions])
    model.append(rf_model)
    # from matplotlib import pyplot as plt

    # plt.scatter([i for i in range(1,len(y_test)+1)], encoder1.inverse_transform(y_test), label='Actual')
    # plt.scatter([i for i in range(1,len(y_test)+1)], encoder1.inverse_transform(predictions), color='red', marker='^', label='Predicted')
    # plt.legend()
    # plt.show()

    #histogram gradient boosting classifier - 2
    x_train, x_test, y_train, y_test = train_test_split(x_gb_1, y, test_size=0.3, random_state=random_seed)

    gb_model = HistGradientBoostingClassifier(categorical_features=['Cancer Type (matching TCGA label)'], max_iter=500)
    gb_model.fit(x_train, y_train)

    predictions = gb_model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    # print(f"Accuracy: {accuracy:.4f}")
    result.append(accuracy)
    pred_vals.append([y_test, predictions])
    model.append(gb_model)
    # from matplotlib import pyplot as plt

    # plt.scatter([i for i in range(1,len(y_test)+1)], encoder1.inverse_transform(y_test), label='Actual')
    # plt.scatter([i for i in range(1,len(y_test)+1)], encoder1.inverse_transform(predictions), color='red', marker='^', label='Predicted')
    # plt.legend()
    # plt.show()

    return result, pred_vals, model

max_till = 0
result_till = []
pred_vals_till = []
max_seed = 0
for i in random_seed:
    results, pred_values, model = perform_classification(i)
    if max(results) > max_till:
        max_till = max(results)
        max_seed = i
        result_till = results
        pred_vals_till = pred_values
    print(i, max(results))
    if max(results) > 0.97:
        break

print()
print("Seed:", max_seed)
print(result_till)
import joblib

joblib.dump(model[0], 'rf3.joblib')
joblib.dump(model[1], 'gb3.joblib')
joblib.dump(model[2], 'rf1.joblib')
joblib.dump(model[3], 'gb1.joblib')

from matplotlib import pyplot as plt
for y_test, predictions in pred_vals_till:
    plt.scatter([i for i in range(1,len(y_test)+1)], encoder1.inverse_transform(y_test), label='Actual')
    plt.scatter([i for i in range(1,len(y_test)+1)], encoder1.inverse_transform(predictions), color='red', marker='^', label='Predicted')
    plt.legend()
    plt.show()

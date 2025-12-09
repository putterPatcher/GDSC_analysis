from path import relative_dir, table_6

path = relative_dir+table_6

import pandas as pd
df = pd.read_csv(path)

from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

print(df.isna().sum())
df.dropna(inplace=True)
print(df.isna().sum())

x = df.iloc[:, 1:]
encoder = LabelEncoder()
y = encoder.fit_transform(df.iloc[:, 0])

# Create a Gaussian Naive Bayes classifier
gnb = GaussianNB()

# Train the classifier
gnb.fit(x, y)

# Make predictions on the test set
y_pred = gnb.predict(x)

# Evaluate the classifier
accuracy = metrics.accuracy_score(y, y_pred)
print("Accuracy:", accuracy)

import joblib

joblib.dump(gnb, 'table_6.joblib')

from matplotlib import pyplot as plt

plt.scatter([i for i in range(1,len(y)+1)], y, label='Actual')
plt.scatter([i for i in range(1,len(y)+1)], y_pred, color='red', marker='^', label='Predicted')
plt.ylabel('Cancer Cell Line')
plt.legend()
plt.show()

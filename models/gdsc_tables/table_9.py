from path import relative_dir, table_9

path = relative_dir + table_9

import pandas as pd

df_dic = pd.read_excel(path, sheet_name=None)
for sheet_name, df in df_dic.items():
    print("Sheetname:", sheet_name)
    if sheet_name == 'caption':
        continue
    from sklearn.naive_bayes import BernoulliNB
    from sklearn import metrics
    from sklearn.preprocessing import LabelEncoder

    x = df.iloc[:, 1:]
    encoder = LabelEncoder()
    y = encoder.fit_transform(df.iloc[:, 0])

    # Create a Gaussian Naive Bayes classifier
    gnb = BernoulliNB()

    # Train the classifier
    gnb.fit(x, y)

    # Make predictions on the test set
    y_pred = gnb.predict(x)

    # Evaluate the classifier
    accuracy = metrics.accuracy_score(y, y_pred)
    print("Accuracy:", accuracy)

    import joblib

    joblib.dump(gnb, 'table_9_{}.joblib'.format(sheet_name))

    from matplotlib import pyplot as plt

    plt.scatter([i for i in range(1,len(y)+1)], y, label='Actual')
    plt.scatter([i for i in range(1,len(y)+1)], y_pred, color='red', marker='^', label='Predicted')
    plt.ylabel('Drug')
    plt.legend()
    plt.show()

with open('gdsc.json', 'r') as file:
    import json
    y_test, y_pred = json.loads(file.readlines()[0])
    import matplotlib.pyplot as plt
    plt.scatter(y_test, y_pred, color='green', marker='^')
    plt.ylabel("Predicted")
    plt.xlabel("Actual")
    plt.show()
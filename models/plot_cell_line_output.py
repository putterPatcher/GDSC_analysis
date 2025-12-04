with open('cell_line_output.txt') as file:
    data = file.readlines()
    data = data[12:-3]
    x = []
    y = []
    for i in data:
        X, Y = i.split(' ')
        x.append(int(X))
        y.append(float(Y))
    from matplotlib import pyplot as plt
    plt.plot(x, y)
    plt.xlabel("Seed")
    plt.ylabel("Accuracy")
    plt.title("Accuracy v/s Seed")
    plt.show()
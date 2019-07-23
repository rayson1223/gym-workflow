import sys
import matplotlib.pyplot as plt
import csv
import json


def main():
    data = []
    with open('./data/weibull-data.csv') as f:
        reader = csv.reader(f)
        for line in reader:
            data = line
    for k, v in enumerate(data):
        v = v.replace('[', '')
        v = v.replace(']', '')
        data[k] = float(v)
    # print(data)
    data.sort()
    plt.figure(figsize=(50, 10))
    plt.plot(data)
    plt.show()


if __name__ == '__main__':
    sys.exit(main())

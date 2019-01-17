import matplotlib.pyplot as plt
import numpy as np
import sys, os, csv


def line_graph(x, y, x_title='', y_title='', title='', output=None):
    plt.plot(x, y)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)
    if output is not None:
        plt.savefig(output)
    plt.show()


def main():
    # Check input and output path is existence
    if len(sys.argv) < 2:
        raise FileNotFoundError("No input detected!")
    elif len(sys.argv) < 3:
        raise FileNotFoundError("No output path provided!")

    # Check input existence
    if not os.path.exists(sys.argv[1]):
        raise FileNotFoundError("File not exist!")

    input_source = sys.argv[1]
    output = sys.argv[2]
    cluster_size_data = {}
    exec_time_data = list()

    # Import data
    with open(input_source, 'r') as csvfile:
        data = csv.DictReader(csvfile)
        next(data)  # skip header
        for row in data:
            exec_time_data.append(float(row['exec_time']))
            if row['cluster_size'] in cluster_size_data.keys():
                cluster_size_data[row['cluster_size']].append(float(row['exec_time']))
            else:
                cluster_size_data[row['cluster_size']] = list()

    # Default Data Preparation
    x_axis = list(range(1, len(exec_time_data) + 1))  # define x-axis as episode based

    # Begin process graphs
    line_graph(x_axis, exec_time_data, 'Episodes', 'Execution Time(s)', 'Overall Execution Time by episode', output)


if __name__ == '__main__':
    sys.exit(main())

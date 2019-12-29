import sys
import csv
import os


def main():
    data = {}
    with open('./data/fundamental/final_cn_result_30.csv') as f:
        reader = csv.DictReader(f)
        for line in reader:
            degrees = float(line['degrees'])
            # cs = int(line['cluster_size'])
            cn = int(line['cluster_num'])
            wall_time = float(line['wall_time'])
            if degrees not in data:
                data[degrees] = {}
            # Cluster Size operation
            # if cs not in data[degrees]:
            #     data[degrees][cs] = []
            # data[degrees][cs].append(wall_time)

    #         Cluster num operation
            if cn not in data[degrees]:
                data[degrees][cn] = []
            data[degrees][cn].append(wall_time)

    with open("gen.txt", 'w') as f:
        f.write("{\n")
        for degree, cp_data in data.items():
            f.write("{}: {{\n".format(degree))
            for cp, d in cp_data.items():
                f.write("{}: lambda: random.randrange({}, {}, 1),\n".format(cp, min(d), max(d)))
            f.write("},")
        f.write("}")


if __name__ == '__main__':
    sys.exit(main())

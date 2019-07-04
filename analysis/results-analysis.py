import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import os
import agents.utils.plotting as draw
import json
from collections import namedtuple

csv.field_size_limit(sys.maxsize)


def getCsvTotalLine(filename):
    with open(filename) as f:
        return sum(1 for line in csv.reader(f, delimiter="\n"))


def convertSingleValueCsvToList(filename):
    x = list()
    with open(filename) as f:
        reader = csv.reader(f, delimiter='\n')
        for line in reader:
            x.append(float(line[0].split(',')[1]))
    return x


def convertJsonValueCsvToList(filename):
    x = list()
    with open(filename) as f:
        reader = csv.DictReader(f)
        next(reader)
        for line in reader:
            x.append(line['records'])
    return x


def main():
    # Check input and output path is existence
    if len(sys.argv) < 5:
        raise FileNotFoundError(
            "Insufficient inputs! Please follow the file structure below " +
            "\n (episode_length, reward, total_reward, exec_records)"
        )
    if not os.path.exists(sys.argv[1]):
        raise FileNotFoundError("Episode Length File not exist!")
    if not os.path.exists(sys.argv[2]):
        raise FileNotFoundError("Episode Reward File not exist!")
    if not os.path.exists(sys.argv[3]):
        raise FileNotFoundError("Episode Total Reward File not exist!")
    if not os.path.exists(sys.argv[4]):
        raise FileNotFoundError("Training Exec Record not exist!")

    epi_length = sys.argv[1]
    epi_reward = sys.argv[2]
    epi_total_reward = sys.argv[3]
    exec_records = sys.argv[4]

    # Reconstruct training stats data format
    EpisodeStats = namedtuple("Stats",
                              ["episode_lengths", "episode_rewards", "episode_total_reward"])
    num_episodes = getCsvTotalLine(epi_length)
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        episode_total_reward=np.zeros(num_episodes),
    )
    stats = stats._replace(episode_lengths=convertSingleValueCsvToList(epi_length))
    stats = stats._replace(episode_rewards=convertSingleValueCsvToList(epi_reward))
    stats = stats._replace(episode_total_reward=convertSingleValueCsvToList(epi_total_reward))
    all_epi_exec_record = convertJsonValueCsvToList(exec_records)

    box_x = list()
    for record in all_epi_exec_record:
        record_json = json.loads(record)
        box_x.append(record_json['overhead'])
    draw.plot_boxplot(box_x, labels=list(range(1, len(box_x)+1)), xlabel="Episode", ylabel="Overhead", )
    draw.plot_episode_stats(stats)


if __name__ == '__main__':
    sys.exit(main())

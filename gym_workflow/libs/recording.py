import csv
import os


def write_record(
        data,
        header=[],
        filename="episode_stats.csv"
):
    if not os.path.exists(os.getcwd() + "/records/" + filename):
        with open(os.getcwd() + "/records/" + filename, 'w', newline='', encoding='utf-8') as r:
            writer = csv.DictWriter(r, fieldnames=header)
            writer.writeheader()

    with open(os.getcwd() + "/records/" + filename, 'a') as r:
        writer = csv.writer(r)
        writer.writerow(data)


def write_episode(
        data,
        header=[
            'clusters', 'action', 'status', 'exec_time',
            'wall_time', 'cum_wall_time', 'best_record',
            'improvment' 'reward'
        ],
        file_name="workflow_record.csv"
):
    if not os.path.exists(os.getcwd() + "/records/" + file_name):
        with open(os.getcwd() + "/records/" + file_name, 'w', newline='', encoding='utf-8') as r:
            writer = csv.DictWriter(r, fieldnames=header)
            writer.writeheader()

    with open(os.getcwd() + "/records/" + file_name, 'a') as r:
        writer = csv.writer(r)
        writer.writerow(data)


def write_training_status(
        data,
        header=['episode', 'Q', 'stats', 'action', 'action_prob', 'reward'],
        file_name="v5_training_records.csv"
):
    if not os.path.exists(os.getcwd() + "/records/" + file_name):
        with open(os.getcwd() + "/records/" + file_name, 'w', newline='', encoding='utf-8') as r:
            writer = csv.DictWriter(r, fieldnames=header)
            writer.writeheader()

    with open(os.getcwd() + "/records/" + file_name, 'a') as r:
        writer = csv.writer(r)
        writer.writerow(data)

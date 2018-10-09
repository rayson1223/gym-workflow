import csv
import os


def write_episode(
	data,
	header=[
		'clusters', 'action', 'status', 'exec_time',
		'wall_time', 'cum_wall_time', 'best_record',
		'improvment' 'reward'
	],
	file_name="v5_record.csv"
):
	if not os.path.exists(os.getcwd() + "records/" + file_name):
		with open(os.getcwd() + "records/" + file_name, 'w', newline='', encoding='utf-8') as r:
			writer = csv.DictWriter(r, fieldnames=header)
			writer.writeheader()
	
	with open(os.getcwd() + "records/" + file_name, 'a') as r:
		writer = csv.writer(r)
		writer.writerow(data)

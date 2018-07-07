import csv
import os
from gym_workflow.lib.montage.db.model.pegasus_wf import PegasusWf


def main():
	with open('workflow_record.csv', newline='') as f:
		reader = csv.reader(f)
		header = next(reader)

		# Create results file if there are no results recorded
		if not os.path.exists(os.getcwd() + "/results.csv"):
			with open(os.getcwd() + "/results.csv", 'w', newline='', encoding='utf-8') as r:
				fieldnames = [
					'submit_dir', 'cluster_size', 'cluster_num', 'exec_time', 'wall_time',
					'cum_wall_time'
				]
				writer = csv.DictWriter(r, fieldnames=fieldnames)
				writer.writeheader()

		with open('results.csv', 'a') as r:
			writer = csv.writer(r)
			for row in reader:
				submit_dir, cs, cn = row
				wf = PegasusWf()
				wf.initialize_by_work_dir(submit_dir)
				writer.writerow([submit_dir, cs, cn, wf.get_jobs_run_by_time(), wf.get_wall_time(), wf.get_cum_time()])


if __name__ == "__main__":
	main()

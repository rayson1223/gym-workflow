import csv
import os
from gym_workflow.lib.montage.db.model.pegasus_wf import PegasusWf
import sys


def main(input="workflow_record.csv", output="result.csv"):
	if len(sys.argv) == 2:
		input = sys.argv[1]
	elif len(sys.argv) == 3:
		input = sys.argv[1]
		output = sys.argv[2]
	
	with open(input, newline='') as f:
		reader = csv.reader(f)
		header = next(reader)

		# Create results file if there are no results recorded
		if not os.path.exists(os.getcwd() + "/{}".format(output)):
			with open(os.getcwd() + "/{}".format(output), 'w', newline='', encoding='utf-8') as r:
				fieldnames = [
					'degrees', 'submit_dir', 'cluster_size', 'cluster_num', 'exec_time', 'wall_time',
					'cum_wall_time'
				]
				writer = csv.DictWriter(r, fieldnames=fieldnames)
				writer.writeheader()
		
		with open(output, 'a') as r:
			writer = csv.writer(r)
			for row in reader:
				degrees, submit_dir, cs, cn = row
				wf = PegasusWf()
				wf.initialize_by_work_dir(submit_dir)
				# Process output folder
				tmp = submit_dir.split('/')
				folder_name = tmp[len(tmp) - 1]
				
				# Plot the dax
				wf.dax_plot("~/experiment-dax/{}".format(folder_name))
				
				# Write record to result output file
				writer.writerow(
					[degrees, submit_dir, cs, cn, wf.get_jobs_run_by_time(), wf.get_wall_time(), wf.get_cum_time()])


if __name__ == "__main__":
	main()

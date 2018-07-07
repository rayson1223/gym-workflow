from gym_workflow.lib.montage.montage import Montage
import time


def main():
	for cs in range(10):
		for cn in range(10):
			a = Montage()
			a.build(cs + 1, cn + 1)
			a.pegasus_run()

			# Wait for the job submission status
			time.sleep(5)
			a.write_record(cs + 1, cn + 1)

	# a.pegasus_remove()


if __name__ == "__main__":
	main()

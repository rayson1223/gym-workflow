from gym_workflow.lib.montage.montage import Montage
from gym_workflow.envs.database import DatabaseEnv
import time


def main():
	db_dir = ".pegasus/workflow.db"
	for cs in range(10):
		for cn in range(10):
			a = Montage()
			a.build_transformation_catalog(cs + 1, cn + 1)
			a.generate_region_hdr()
			a.process_color_band()
			a.write_rc()
			a.write_property_conf()
			a.pegasus_plan()
			a.pegasus_run()
			db = DatabaseEnv(db_dir)
			# Wait for the job submission status
			time.sleep(10)
			a.write_record(cs + 1, cn + 1)
			db.observe(a.work_dir)
		# a.pegasus_remove()


if __name__ == "__main__":
	main()

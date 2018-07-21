import subprocess
from gym_workflow.lib.montage.db.queries import SQLDb


class PegasusWf:
	def __init__(self, master_db_url=".pegasus/workflow.db", wf_id=None):
		self.master_db_url = master_db_url
		self.wf_id = None
		self.root_wf_uuid = None
		self.wf_db_url = None
		self.stampede_db = None
		try:
			home_dir = subprocess.getoutput("cd ~; pwd")
			self._session = SQLDb("%s/%s" % (home_dir, self.master_db_url))
		except Error as e:
			raise e

		# If the ID is specified, it means that the query is specific to a workflow.
		# So we will now query the master database to get the connection URL for the workflow.
		if wf_id:
			self.initialize(wf_id)

	def initialize(self, wf_id):
		try:
			self.wf_id, self.root_wf_uuid, self.wf_db_url = self._session.getone(
				"select wf_id, wf_uuid, db_url from master_workflow where wf_id=%s or wf_uuid=%s" % (wf_id, wf_id)
			)
			if self.wf_db_url:
				db_loc = self.wf_db_url.split('sqlite:///')[1]
				self.stampede_db = SQLDb(db_loc)
		except Exception as e:
			print("Initializing: " + e.__str__())

	def initialize_by_work_dir(self, work_dir):
		try:
			self.wf_id, self.root_wf_uuid, self.wf_db_url = self._session.getone(
				"select wf_id, wf_uuid, db_url from master_workflow where submit_dir='%s'" % work_dir
			)
			if self.wf_db_url:
				db_loc = self.wf_db_url.split('sqlite:///')[1]
				self.stampede_db = SQLDb(db_loc)
		except Exception as e:
			print("Initializing by work dir: " + e.__str__())

	def get_wall_time(self):
		def get_workflow_wall_time(workflow_states_list):
			"""
			Utility method for returning the workflow wall time given all the workflow states
			@worklow_states_list list of all workflow states.
			"""
			workflow_wall_time = None
			workflow_start_event_count = 0
			workflow_end_event_count = 0
			is_end = False
			workflow_start_cum = 0
			workflow_end_cum = 0
			for workflow_state in workflow_states_list:
				id, state, timestamp, restart_count, status, reason = workflow_state
				if state == 'WORKFLOW_STARTED':
					workflow_start_event_count += 1
					workflow_start_cum += timestamp
				else:
					workflow_end_event_count += 1
					workflow_end_cum += timestamp
			if workflow_start_event_count > 0 and workflow_end_event_count > 0:
				if workflow_start_event_count == workflow_end_event_count:
					workflow_wall_time = workflow_end_cum - workflow_start_cum
			return workflow_wall_time

		if self.stampede_db:
			return get_workflow_wall_time(self.stampede_db.getall(
				"select * from main.workflowstate where wf_id = 1"
			))
		else:
			return None

	def get_cum_time(self):
		if self.stampede_db:
			return self.stampede_db.getone(
				"select sum(remote_duration * multiplier_factor) FROM invocation as invoc, job_instance as ji WHERE invoc.task_submit_seq >= 0 and invoc.job_instance_id = ji.job_instance_id and invoc.wf_id in (1) and invoc.transformation <> 'condor::dagman'"
			)[0]
		else:
			return None

	def get_jobs_run_by_time(self):
		def sum_all_time(jl):
			total_runtime = 0
			for t in jl:
				date_format, count, rt = t
				total_runtime += rt
			return total_runtime

		if self.stampede_db:
			return sum_all_time(self.stampede_db.getall(
				"""
					select (js.timestamp/ 2629743) as date_format,count(ji.job_instance_id) as count,
					sum(ji.local_duration) as total_runtime 
					from
						workflow wi,
						job j,
						job_instance  ji,
						jobstate js
					where wi.root_wf_id = 1
						and wi.wf_id=j.wf_id
						and j.job_id=ji.job_id
						and js.job_instance_id = ji.job_instance_id
						and js.state = 'EXECUTE'
					group by date_format
					order by date_format
				"""
			))
		else:
			return None

	def get_job_distribution_stat(self):
		return self.stampede_db.getall(
			"""
				SELECT transformation,
				count(invocation_id) as count,
				min(remote_duration * multiplier_factor) as min,
				count(CASE WHEN (invoc.exitcode = 0 and invoc.exitcode is NOT NULL) THEN invoc.exitcode END) AS success,
				count(CASE WHEN (invoc.exitcode != 0 and invoc.exitcode is NOT NULL) THEN invoc.exitcode END) AS failure,
				max(remote_duration * multiplier_factor) as max,
				avg(remote_duration * multiplier_factor) as avg,
				sum(remote_duration * multiplier_factor) as sum FROM
				invocation as invoc, job_instance as ji WHERE
				invoc.job_instance_id = ji.job_instance_id and
				invoc.wf_id IN (1) GROUP BY transformation
			"""
		)


def main():
	wf = PegasusWf()
	wf.initialize_by_work_dir("/Users/rayson/PycharmProjects/gym-workflow/gym_workflow/lib/montage/work/1530199580407")
	# print(wf.wf_db_url)
	print(wf.get_wall_time())
	print(wf.get_cum_time())
	print(wf.get_jobs_run_by_time())


if __name__ == "__main__":
	main()
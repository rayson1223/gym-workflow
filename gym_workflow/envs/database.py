from sqlite3 import connect, Error
from time import sleep


# from karellen.sqlite3 import Connection, UpdateHookOps


class DatabaseEnv:
	# ONLY support pegasus default SQLite DB
	PEGASUS_WF_STATUS = {0: 'Complete', 2: 'Aborted'}

	def __init__(self, db_dir):
		self.dbDir = db_dir
		self.observe_wf = 0
		self.is_done = False
		self.conn = connect(self.dbDir)

	# self.conn = connect(self.dbDir, factory=Connection)
	# self.conn.set_update_hook(self.callback_hook)

	# @staticmethod
	# def callback_hook(conn, op, db_name, table_name, rowid):
	# 	"""Handle notification here. Do not modify the connection!"""
	# 	if op == UpdateHookOps.SQLITE_INSERT and self.observe_wf == rowid and table_name == "master_workflowstate":
	# 		self.is_done = True
	# 	s = '%s %ld of table "%s" in database "%s"' % (op, rowid, table_name, db_name)
	# 	print(s)

	def observe(self, wf_id):
		self.observe_wf = wf_id
		while True:
			res = self.getone(
				"select * from master_workflowstate where wf_id=%s and state='WORKFLOW_TERMINATED';" % self.observe_wf)
			if res is None:
				print("No complete signal received yet!")
				sleep(10)
			else:
				print("Workflow Complete with the state of %s" % self.PEGASUS_WF_STATUS[res[4]])
				break

	def getone(self, cmd):
		try:
			return self.conn.cursor().execute(cmd).fetchone()
		except Error as e:
			print("An error occurred:", e.args[0])

	def exec(self, cmd):
		try:
			self.conn.execute(cmd)
			self.conn.commit()
		except Error as e:
			print("An error occurred:", e.args[0])

	def close(self):
		self.conn.close()


# Example Script
# def main():
# 	db = DatabaseEnv("/Users/rayson/Documents/master/experiments/db/workflow.db")
# 	db.observe(22)
#
# if __name__ == '__main__':
# 	main()

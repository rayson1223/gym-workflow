from sqlite3 import connect, Error


class SQLDb:
	def __init__(self, db_dir):
		try:
			self._session = connect(db_dir)
		except Error as e:
			raise e

	def close(self):
		self._session.close()

	def getone(self, cmd):
		try:
			return self._session.cursor().execute(cmd).fetchone()
		except Error as e:
			print("An error occurred:", e.args[0])

	def getall(self, cmd):
		try:
			return self._session.cursor().execute(cmd).fetchall()
		except Error as e:
			print("An error occurred:", e.args[0])

	def exec(self, cmd):
		try:
			self._session.execute(cmd)
			self._session.commit()
		except Error as e:
			print("An error occurred:", e.args[0])

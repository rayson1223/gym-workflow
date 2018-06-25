import calendar, time, sys, os, re, subprocess
from astropy.io import ascii
from gym_workflow.lib.montage.auto_adag import *
from gym_workflow.lib.pegasus.DAX3 import *
import random


class Montage:
	common_files = {}
	replica_catalog = {}

	def __init__(
			self, type="2mass", center="15.09552 -0.74559",
			degrees=0.1, band=["2mass:j:red"], target="regular"
	):
		self.montage_type = type
		self.center = center  # Center of the output, for example M17 or 56.5 23.75
		self.degrees = degrees  # Number of degrees of side of the output
		self.band = band  # Band definition. Example: [dss:DSS2B:red]
		self.tc_target = target
		self.data_dir = os.getcwd() + "/data"
		self.work_dir = os.getcwd() + "/work"
		if os.path.exists(self.data_dir):
			print("data directory already exists")
		else:
			os.mkdir(self.data_dir)
			print("data directory created")
		if os.path.exists(self.work_dir):
			print("work directory already exists")
		else:
			os.mkdir(self.work_dir)
			print("work directory created")

		self.folder_name = int(round(time.time() * 1000))  # calendar.timegm(time.gmtime())
		self.data_dir = "%s/%s" % (self.data_dir, self.folder_name)
		self.work_dir = "%s/%s" % (self.work_dir, self.folder_name)

		if not os.path.exists(self.data_dir):
			os.mkdir(self.data_dir)
		if not os.path.exists(self.work_dir):
			os.mkdir(self.work_dir)

		self.cs = 1
		self.cn = 1

		self.dax = AutoADAG("montage")

	# Setup Mail Notification
	# share_dir = \
	# subprocess.Popen("pegasus-config --sh-dump | grep ^PEGASUS_SHARE_DIR= | sed -e 's/.*=//' -e 's/\"//g'", shell=True, stdout=subprocess.PIPE).communicate()[0]
	# share_dir = share_dir.strip()
	# self.dax.invoke('start', share_dir + "/notification/email")
	# self.dax.invoke('on_error', share_dir + "/notification/email")
	# self.dax.invoke('on_success', share_dir + "/notification/email --report=pegasus-statistics")

	def build_transformation_catalog(self, clusters_size=1, clusters_num=1):
		"""
			Some transformations in Montage uses multiple executables
		"""
		exes = {}
		self.cs = clusters_size
		self.cn = clusters_num
		full_path = subprocess.getoutput("which mProject")
		if full_path is None:
			raise RuntimeError("mProject is not in the $PATH")
		base_dir = os.path.dirname(full_path)

		f = open(self.data_dir + "/tc.txt", "w")
		if self.tc_target == "container":
			f.write("cont montage {\n")
			f.write("   type \"singularity\"\n")
			f.write("   image \"shub://pegasus-isi/montage-workflow-v2\"\n")
			f.write("   profile env \"MONTAGE_HOME\" \"/opt/Montage\"\n")
			f.write("}\n")

		for fname in os.listdir(base_dir):
			if fname[0] == ".":
				continue
			if fname[0] == "mDiffFit":
				# special compound transformation - see below
				continue
			f.write("\n")
			f.write("tr %s {\n" % (fname))
			if self.tc_target == "regular":
				f.write("  site local {\n")
				f.write("    type \"STAGEABLE\"\n")
				f.write("    arch \"x86_64\"\n")
				f.write("    pfn \"file://%s/%s\"\n" % (base_dir, fname))
			else:
				# container
				f.write("  site condor_pool {\n")
				f.write("    type \"INSTALLED\"\n")
				f.write("    container \"montage\"\n")
				f.write("    pfn \"file://%s/%s\"\n" % (base_dir, fname))

			# Control what kind of jobs use clustering
			if fname in ["mProject", "mDiff", "mDiffFit", "mBackground"]:
				# TODO: Customize the clustering by configuration
				# Horizontal Clustering
				# 1) clusters.size factor
				#
				# The clusters.size factor denotes how many jobs need to be merged into a single clustered job.
				# It is specified via the use of a PEGASUS namespace profile key 'clusters.size'.
				# for e.g.
				#   if at a particular level, say 4 jobs referring to logical transformation B have been scheduled to a siteX.
				#   The clusters.size factor associated with job B for siteX is say 3. This will result in 2 clustered jobs,
				#   one composed of 3 jobs and another of 2 jobs.
				f.write("    profile pegasus \"clusters.size\" \"%s\"\n" % clusters_size)

			# 2) clusters.num factor
			#
			# The clusters.num factor denotes how many clustered jobs does the user want to see per level per site.
			# It is specified via the use of a PEGASUS namespace profile key 'clusters.num'. for e.g.
			#
			#   if at a particular level, say 4 jobs referring to logical transformation B have been scheduled to a siteX.
			#   The 'clusters.num' factor associated with job B for siteX is say 3.
			#   This will result in 3 clustered jobs, one composed of 2 jobs and others of a single job each.

			# Runtime clustering
			# f.write("    profile pegasus \"runtime\" \"100\"\n")
			f.write("  }\n")
			f.write("}\n")
		f.close()

		# some Montage tools depend on other tools
		for tname in ["mDiffFit"]:
			t = Transformation(tname)
			if tname == "mDiffFit":
				t.uses(Executable("mDiff"))
				t.uses(Executable("mFitplane"))
			t.uses(Executable("mDiffFit"))
			self.dax.addTransformation(t)

	def generate_region_hdr(self):
		global common_files

		(crval1, crval2) = self.center.split()
		crval1 = float(crval1)
		crval2 = float(crval2)

		cdelt = 0.000277778
		naxis = int((float(self.degrees) / cdelt) + 0.5)
		crpix = (naxis + 1) / 2.0

		f = open("%s/region.hdr" % self.data_dir, "w")
		f.write("SIMPLE  = T\n")
		f.write("BITPIX  = -64\n")
		f.write("NAXIS   = 2\n")
		f.write("NAXIS1  = %d\n" % (naxis))
		f.write("NAXIS2  = %d\n" % (naxis))
		f.write("CTYPE1  = 'RA---TAN'\n")
		f.write("CTYPE2  = 'DEC--TAN'\n")
		f.write("CRVAL1  = %.6f\n" % (crval1))
		f.write("CRVAL2  = %.6f\n" % (crval2))
		f.write("CRPIX1  = %.6f\n" % (crpix))
		f.write("CRPIX2  = %.6f\n" % (crpix))
		f.write("CDELT1  = %.9f\n" % (-cdelt))
		f.write("CDELT2  = %.9f\n" % (cdelt))
		f.write("CROTA2  = %.6f\n" % (0.0))
		f.write("EQUINOX = %d\n" % (2000))
		f.write("END\n")
		f.close()

		self.common_files["region.hdr"] = File("region.hdr")
		self.replica_catalog["region.hdr"] = {"url": "file://" + self.data_dir + "/region.hdr", "site_label": "local"}

		# we also need an oversized region which will be used in the first part of the
		# workflow to get the background correction correct
		f = open("%s/region-oversized.hdr" % self.data_dir, "w")
		f.write("SIMPLE  = T\n")
		f.write("BITPIX  = -64\n")
		f.write("NAXIS   = 2\n")
		f.write("NAXIS1  = %d\n" % (naxis + 3000))
		f.write("NAXIS2  = %d\n" % (naxis + 3000))
		f.write("CTYPE1  = 'RA---TAN'\n")
		f.write("CTYPE2  = 'DEC--TAN'\n")
		f.write("CRVAL1  = %.6f\n" % (crval1))
		f.write("CRVAL2  = %.6f\n" % (crval2))
		f.write("CRPIX1  = %.6f\n" % (crpix + 1500))
		f.write("CRPIX2  = %.6f\n" % (crpix + 1500))
		f.write("CDELT1  = %.9f\n" % (-cdelt))
		f.write("CDELT2  = %.9f\n" % (cdelt))
		f.write("CROTA2  = %.6f\n" % (0.0))
		f.write("EQUINOX = %d\n" % (2000))
		f.write("END\n")
		f.close()

		self.common_files["region-oversized.hdr"] = File("region-oversized.hdr")
		self.replica_catalog["region-oversized.hdr"] = \
			{"url": "file://" + self.data_dir + "/region-oversized.hdr", "site_label": "local"}

	def process_color_band(self):
		band_id = 0
		color_band = {}
		for band_def in self.band:
			band_id += 1
			(survey, band, color) = band_def.split(":")
			self.add_band(band_id, survey, band, color)
			color_band[color] = band_id

		# if we have 3 bands in red, blue, green, try to create a color jpeg
		if 'red' in color_band and 'green' in color_band and 'blue' in color_band:
			self.color_jpg(color_band['red'], color_band['green'], color_band['blue'])

	def add_band(self, band_id, survey, band, color):

		band_id = str(band_id)

		# print("\nAdding band %s (%s %s -> %s)" % (band_id, survey, band, color))

		# data find - go a little bit outside the box - see mExec implentation
		# degrees_datafind = str(float(degrees) * 1.42)
		degrees_datafind = str(float(self.degrees))
		cmd = "mArchiveList %s %s \"%s\" %s %s %s/%s-images.tbl" % (
			survey, band, self.center, degrees_datafind, degrees_datafind, self.data_dir, band_id
		)
		# print("Running sub command: " + cmd)
		if subprocess.call(cmd, shell=True) != 0:
			print("Command failed!")
			sys.exit(1)
		self.replica_catalog["%s-images.tbl" % (band_id)] = {
			"url": "file://%s/%s-images.tbl" % (self.data_dir, band_id), "site_label": "local"
		}

		# image tables
		raw_tbl = File("%s-raw.tbl" % (band_id))
		self.replica_catalog[raw_tbl.name] = \
			{"url": "file://%s/%s" % (self.data_dir, raw_tbl.name), "site_label": "local"}
		projected_tbl = File("%s-projected.tbl" % (band_id))
		self.replica_catalog[projected_tbl.name] = \
			{"url": "file://%s/%s" % (self.data_dir, projected_tbl.name), "site_label": "local"}
		corrected_tbl = File("%s-corrected.tbl" % (band_id))
		self.replica_catalog[corrected_tbl.name] = \
			{"url": "file://%s/%s" % (self.data_dir, corrected_tbl.name), "site_label": "local"}
		cmd = "cd %s && mDAGTbls %s-images.tbl region-oversized.hdr %s %s %s" % (
			self.data_dir, band_id, raw_tbl.name, projected_tbl.name, corrected_tbl.name
		)
		# print("Running sub command: " + cmd)
		if subprocess.call(cmd, shell=True) != 0:
			print("Command failed!")
			sys.exit(1)

		# diff table
		cmd = "cd %s && mOverlaps %s-raw.tbl %s-diffs.tbl" % (
			self.data_dir, band_id, band_id
		)
		# print("Running sub command: " + cmd)
		if subprocess.call(cmd, shell=True) != 0:
			print("Command failed!")
			sys.exit(1)

		# statfile table
		t = ascii.read("%s/%s-diffs.tbl" % (self.data_dir, band_id))

		# make sure we have a wide enough column
		t['stat'] = "                                                                  "
		for row in t:
			base_name = re.sub("(diff\.|\.fits.*)", "", row['diff'])
			row['stat'] = "%s-fit.%s.txt" % (band_id, base_name)
		ascii.write(t, "%s/%s-stat.tbl" % (self.data_dir, band_id), format='ipac')
		self.replica_catalog["%s-stat.tbl" % (band_id)] = {
			"url": "file://%s/%s-stat.tbl" % (self.data_dir, band_id), "site_label": "local"
		}

		# for all the input images in this band, and them to the rc, and
		# add reproject tasks
		data = ascii.read("%s/%s-images.tbl" % (self.data_dir, band_id))
		for row in data:
			base_name = re.sub("\.fits.*", "", row['file'])

			# add an entry to the replica catalog
			self.replica_catalog[base_name + ".fits"] = {"url": row['URL'], "site_label": "ipac"}

			# projection job
			j = Job(name="mProject")
			in_fits = File(base_name + ".fits")
			projected_fits = File("p" + base_name + ".fits")
			area_fits = File("p" + base_name + "_area.fits")
			j.uses(self.common_files["region-oversized.hdr"], link=Link.INPUT)
			j.uses(in_fits, link=Link.INPUT)
			j.uses(projected_fits, link=Link.OUTPUT, transfer=False)
			j.uses(area_fits, link=Link.OUTPUT, transfer=False)
			j.addArguments("-X", in_fits, projected_fits, self.common_files["region-oversized.hdr"])
			self.dax.addJob(j)

		fit_txts = []
		data = ascii.read("%s/%s-diffs.tbl" % (self.data_dir, band_id))
		for row in data:
			base_name = re.sub("(diff\.|\.fits.*)", "", row['diff'])

			# mDiffFit job
			j = Job(name="mDiffFit")
			plus = File("p" + row['plus'])
			plus_area = File(re.sub("\.fits", "_area.fits", plus.name))
			minus = File("p" + row['minus'])
			minus_area = File(re.sub("\.fits", "_area.fits", minus.name))
			fit_txt = File("%s-fit.%s.txt" % (band_id, base_name))
			diff_fits = File("%s-diff.%s.fits" % (band_id, base_name))
			j.uses(plus, link=Link.INPUT)
			j.uses(plus_area, link=Link.INPUT)
			j.uses(minus, link=Link.INPUT)
			j.uses(minus_area, link=Link.INPUT)
			j.uses(self.common_files["region-oversized.hdr"], link=Link.INPUT)
			j.uses(fit_txt, link=Link.OUTPUT, transfer=False)
			# j.uses(diff_fits, link=Link.OUTPUT, transfer=True)
			j.addArguments("-d", "-s", fit_txt, plus, minus, diff_fits, self.common_files["region-oversized.hdr"])
			self.dax.addJob(j)
			fit_txts.append(fit_txt)

		# mConcatFit
		j = Job(name="mConcatFit")
		stat_tbl = File("%s-stat.tbl" % (band_id))
		j.uses(stat_tbl, link=Link.INPUT)
		j.addArguments(stat_tbl)
		fits_tbl = File("%s-fits.tbl" % (band_id))
		j.uses(fits_tbl, link=Link.OUTPUT, transfer=False)
		j.addArguments(fits_tbl)
		for fit_txt in fit_txts:
			j.uses(fit_txt, link=Link.INPUT)
		j.addArguments(".")
		self.dax.addJob(j)

		# mBgModel
		j = Job(name="mBgModel")
		j.addArguments("-i", "100000")
		images_tbl = File("%s-images.tbl" % (band_id))
		j.uses(images_tbl, link=Link.INPUT)
		j.addArguments(images_tbl)
		j.uses(fits_tbl, link=Link.INPUT)
		j.addArguments(fits_tbl)
		corrections_tbl = File("%s-corrections.tbl" % (band_id))
		j.uses(corrections_tbl, link=Link.OUTPUT, transfer=False)
		j.addArguments(corrections_tbl)
		self.dax.addJob(j)

		# mBackground
		data = ascii.read("%s/%s-raw.tbl" % (self.data_dir, band_id))
		for row in data:
			base_name = re.sub("(diff\.|\.fits.*)", "", row['file'])

			# mBackground job
			j = Job(name="mBackground")
			projected_fits = File("p" + base_name + ".fits")
			projected_area = File("p" + base_name + "_area.fits")
			corrected_fits = File("c" + base_name + ".fits")
			corrected_area = File("c" + base_name + "_area.fits")
			j.uses(projected_fits, link=Link.INPUT)
			j.uses(projected_area, link=Link.INPUT)
			j.uses(projected_tbl, link=Link.INPUT)
			j.uses(corrections_tbl, link=Link.INPUT)
			j.uses(corrected_fits, link=Link.OUTPUT, transfer=False)
			j.uses(corrected_area, link=Link.OUTPUT, transfer=False)
			j.addArguments("-t", projected_fits, corrected_fits, projected_tbl, corrections_tbl)
			self.dax.addJob(j)

		# mImgtbl - we need an updated corrected images table because the pixel offsets and sizes need
		# to be exactly right and the original is only an approximation
		j = Job(name="mImgtbl")
		updated_corrected_tbl = File("%s-updated-corrected.tbl" % (band_id))
		j.uses(corrected_tbl, link=Link.INPUT)
		j.uses(updated_corrected_tbl, link=Link.OUTPUT, transfer=False)
		j.addArguments(".", "-t", corrected_tbl, updated_corrected_tbl)
		data = ascii.read("%s/%s-corrected.tbl" % (self.data_dir, band_id))
		for row in data:
			base_name = re.sub("(diff\.|\.fits.*)", "", row['file'])
			projected_fits = File(base_name + ".fits")
			j.uses(projected_fits, link=Link.INPUT)
		self.dax.addJob(j)

		# mAdd
		j = Job(name="mAdd")
		mosaic_fits = File("%s-mosaic.fits" % (band_id))
		mosaic_area = File("%s-mosaic_area.fits" % (band_id))
		j.uses(updated_corrected_tbl, link=Link.INPUT)
		j.uses(self.common_files["region.hdr"], link=Link.INPUT)
		j.uses(mosaic_fits, link=Link.OUTPUT, transfer=True)
		j.uses(mosaic_area, link=Link.OUTPUT, transfer=True)
		j.addArguments("-e", updated_corrected_tbl, self.common_files["region.hdr"], mosaic_fits)
		data = ascii.read("%s/%s-corrected.tbl" % (self.data_dir, band_id))
		for row in data:
			base_name = re.sub("(diff\.|\.fits.*)", "", row['file'])
			corrected_fits = File(base_name + ".fits")
			corrected_area = File(base_name + "_area.fits")
			j.uses(corrected_fits, link=Link.INPUT)
			j.uses(corrected_area, link=Link.INPUT)
		self.dax.addJob(j)

		# mJPEG - Make the JPEG for this channel
		j = Job(name="mJPEG")
		mosaic_jpg = File("%s-mosaic.jpg" % (band_id))
		j.uses(mosaic_fits, link=Link.INPUT)
		j.uses(mosaic_jpg, link=Link.OUTPUT, transfer=True)
		j.addArguments("-ct", "0", "-gray", mosaic_fits, "0s", "99.999%", "gaussian", "-out", mosaic_jpg)
		self.dax.addJob(j)

	def color_jpg(self, red_id, green_id, blue_id):
		red_id = str(red_id)
		green_id = str(green_id)
		blue_id = str(blue_id)

		# mJPEG - Make the JPEG for this channel
		j = Job(name="mJPEG")
		mosaic_jpg = File("mosaic-color.jpg")
		red_fits = File("%s-mosaic.fits" % (red_id))
		green_fits = File("%s-mosaic.fits" % (green_id))
		blue_fits = File("%s-mosaic.fits" % (blue_id))
		j.uses(red_fits, link=Link.INPUT)
		j.uses(green_fits, link=Link.INPUT)
		j.uses(blue_fits, link=Link.INPUT)
		j.uses(mosaic_jpg, link=Link.OUTPUT, transfer=True)
		j.addArguments( \
			"-red", red_fits, "-1s", "99.999%", "gaussian-log", \
			"-green", green_fits, "-1s", "99.999%", "gaussian-log", \
			"-blue", blue_fits, "-1s", "99.999%", "gaussian-log", \
			"-out", mosaic_jpg)
		self.dax.addJob(j)

	def write_rc(self):
		# write out the replica catalog
		fd = open("%s/rc.txt" % self.data_dir, "w")
		for lfn, data in self.replica_catalog.items():
			fd.write("%s \"%s\"  pool=\"%s\"\n" % (lfn, data['url'], data['site_label']))
		fd.close()

		fd = open("%s/montage.dax" % self.data_dir, "w")
		self.dax.writeXML(fd)
		fd.close()

	def write_property_conf(self):
		fd = open("%s/pegasus.properties" % self.data_dir, "w")
		fd.write("pegasus.metrics.app = Montage\n")
		fd.write("pegasus.catalog.transformation      = Text\n")
		fd.write("pegasus.catalog.transformation.file = %s/tc.txt\n" % self.data_dir)
		fd.write("pegasus.catalog.replica      = File\n")
		fd.write("pegasus.catalog.replica.file = %s/rc.txt\n" % self.data_dir)
		fd.write("pegasus.data.configuration = condorio\n")
		fd.write("pegasus.gridstart.arguments = -f\n")
		fd.close()

	def pegasus_plan(self):
		# Run Planning the cmd after our generated dax
		cmd = "pegasus-plan " \
		      "--dir %s " \
		      "--relative-dir %s " \
		      "--conf %s/pegasus.properties " \
		      "--dax %s/montage.dax " \
		      "--sites condor_pool " \
		      "--output-site local --cluster horizontal" % (
			      os.path.dirname(self.work_dir), self.folder_name, self.data_dir, self.data_dir)
		print("Getting Pegasus Plan executing cmd: %s" % cmd)
		if subprocess.call(cmd, shell=True) != 0:
			print("Command failed!")
			sys.exit(1)

	def pegasus_run(self):
		# Executing pegasus-run cmd for executing planned workflow
		cmd = "pegasus-run %s" % self.work_dir
		print("Running Pegasus Run Cmd: %s" % cmd)

		# Temporary disable execution
		# if subprocess.call(cmd, shell=True) != 0:
		# 	print("Command failed!")
		# 	sys.exit(1)
		return self.gen_exec_time()

	def pegasus_remove(self):
		# pegasus-remove the workflow
		cmd = "pegasus-remove %s" % self.work_dir
		print("Running Pegasus Remove Cmd: %s" % cmd)
		if subprocess.call(cmd, shell=True) != 0:
			print("Command failed!")
			sys.exit(1)

	def gen_exec_time(self):
		cs_degree = {
			0.1: {
				1: lambda: random.randrange(430, 461, 2),
				2: lambda: random.randrange(370, 381, 2),
				3: lambda: random.randrange(360, 371, 2),
				4: lambda: random.randrange(340, 361, 2),
				5: lambda: random.randrange(340, 361, 2),
				6: lambda: random.randrange(340, 361, 2),
				7: lambda: random.randrange(340, 361, 2),
			}
		}
		return cs_degree[self.degrees][self.cs]()


def main():
	a = Montage()
	# Can pass (clusters.size, clusters.num) into method
	a.build_transformation_catalog()

	# region.hdr is the template for the output area
	a.generate_region_hdr()
	a.process_color_band()
	a.write_rc()
	a.write_property_conf()
	a.pegasus_plan()
	a.pegasus_run()
	time.sleep(5)
	a.pegasus_remove()


if __name__ == "__main__":
	main()

import calendar
import time
import sys
import os
import re
import subprocess
from astropy.io import ascii
from gym_workflow.libs.montage.auto_adag import *
from gym_workflow.libs.pegasus.DAX3 import *
from gym_workflow.libs.montage.db.model.pegasus_wf import PegasusWf
import random
import csv


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
		if not os.path.exists(self.data_dir):
			os.mkdir(self.data_dir)
		if not os.path.exists(self.work_dir):
			os.mkdir(self.work_dir)
		
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
	
	def build_transformation_catalog(self, clusters_size=1, clusters_num=None):
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
				if clusters_num is not None:
					f.write("    profile pegasus \"clusters.num\" \"%s\"\n" % clusters_num)
			
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
		print("Running sub command: " + cmd)
		while subprocess.call(cmd, shell=True) != 0:
			print("Command failed: mArchieveList!")
			time.sleep(10)
			
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
		print("Running sub command: " + cmd)
		while subprocess.call(cmd, shell=True) != 0:
			print("Command failed: mDagTbl!")
			time.sleep(10)
		
		# diff table
		cmd = "cd %s && mOverlaps %s-raw.tbl %s-diffs.tbl" % (
			self.data_dir, band_id, band_id
		)
		print("Running sub command: " + cmd)
		while subprocess.call(cmd, shell=True) != 0:
			print("Command failed: mOverlaps!")
			time.sleep(10)
		
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
			      os.path.dirname(self.work_dir), self.folder_name, self.data_dir, self.data_dir
		      )
		print("Getting Pegasus Plan executing cmd: %s" % cmd)
		while subprocess.call(cmd, shell=True) != 0:
			print("Command failed on Pegasus Plan!")
			time.sleep(60)
	
	def pegasus_run(self):
		# Executing pegasus-run cmd for executing planned workflow
		cmd = "pegasus-run %s" % self.work_dir
		print("Running Pegasus Run Cmd: %s" % cmd)
		
		while subprocess.call(cmd, shell=True) != 0:
			print("Command failed on Pegasus Run!")
			time.sleep(60)
	
	def pegasus_remove(self):
		# pegasus-remove the workflow
		cmd = "pegasus-remove %s" % self.work_dir
		print("Running Pegasus Remove Cmd: %s" % cmd)
		while subprocess.call(cmd, shell=True) != 0:
			print("Command failed on Pegasus Remove!")
			time.sleep(60)
	
	def pegasus_status(self):
		cmd = "pegasus-status -l %s" % self.work_dir
		return subprocess.getoutput(cmd)
	
	def monitor_experiment_completion(self):
		complete = False
		while True:
			try:
				out = self.pegasus_status()
				target_index = 0
				# Find the done statistic line
				for index, line in enumerate(out.splitlines()):
					if "%DONE" in line:
						target_index = index + 1
				final_line = out.splitlines()[target_index].split()
			
				if float(final_line[7]) == 100.0 or final_line[8].lower() == 'success':
					complete = True
					break
				elif final_line[8].lower() == 'failure':
					break
			except IndexError as e:
				print(e)
				pass
			except Exception as e:
				print(e)
				pass
			time.sleep(300)
		return complete
	
	def get_results(self):
		wf = PegasusWf()
		wf.initialize_by_work_dir(self.work_dir)
		jbt = wf.get_jobs_run_by_time()
		wt = wf.get_wall_time()
		cwt = wf.get_cum_time()
		return jbt, wt, cwt
	
	@staticmethod
	def gen_exec_time(cs, cn):
		cs_degree = {
			0.1: {
				1: {
					1: lambda: random.randrange(390, 547, 1),
					2: lambda: random.randrange(384, 565, 1),
					3: lambda: random.randrange(399, 434, 1),
					4: lambda: random.randrange(396, 451, 1),
					5: lambda: random.randrange(388, 438, 1),
					6: lambda: random.randrange(399, 443, 1),
					7: lambda: random.randrange(399, 439, 1),
					8: lambda: random.randrange(398, 437, 1),
					9: lambda: random.randrange(405, 446, 1),
					10: lambda: random.randrange(400, 438, 1),
					11: lambda: random.randrange(393, 457, 1),
					12: lambda: random.randrange(396, 438, 1),
					13: lambda: random.randrange(398, 456, 1),
					14: lambda: random.randrange(389, 441, 1),
					15: lambda: random.randrange(398, 468, 1),
					16: lambda: random.randrange(392, 434, 1),
					17: lambda: random.randrange(389, 453, 1),
					18: lambda: random.randrange(399, 438, 1),
					19: lambda: random.randrange(388, 446, 1),
					20: lambda: random.randrange(394, 454, 1),
					21: lambda: random.randrange(402, 470, 1),
					22: lambda: random.randrange(385, 441, 1),
					23: lambda: random.randrange(385, 435, 1),
					24: lambda: random.randrange(400, 449, 1),
					25: lambda: random.randrange(392, 452, 1),
					26: lambda: random.randrange(395, 455, 1),
					27: lambda: random.randrange(390, 459, 1),
					28: lambda: random.randrange(387, 441, 1),
					29: lambda: random.randrange(390, 459, 1),
					30: lambda: random.randrange(376, 442, 1),
				},
				2: {
					1: lambda: random.randrange(356, 376, 1),
					2: lambda: random.randrange(351, 375, 1),
					3: lambda: random.randrange(355, 374, 1),
					4: lambda: random.randrange(359, 384, 1),
					5: lambda: random.randrange(357, 384, 1),
					6: lambda: random.randrange(352, 374, 1),
					7: lambda: random.randrange(360, 377, 1),
					8: lambda: random.randrange(362, 377, 1),
					9: lambda: random.randrange(355, 376, 1),
					10: lambda: random.randrange(357, 379, 1),
					11: lambda: random.randrange(365, 375, 1),
					12: lambda: random.randrange(359, 373, 1),
					13: lambda: random.randrange(356, 384, 1),
					14: lambda: random.randrange(365, 379, 1),
					15: lambda: random.randrange(359, 382, 1),
					16: lambda: random.randrange(356, 386, 1),
					17: lambda: random.randrange(350, 379, 1),
					18: lambda: random.randrange(357, 380, 1),
					19: lambda: random.randrange(358, 375, 1),
					20: lambda: random.randrange(357, 377, 1),
					21: lambda: random.randrange(362, 382, 1),
					22: lambda: random.randrange(356, 382, 1),
					23: lambda: random.randrange(358, 376, 1),
					24: lambda: random.randrange(358, 378, 1),
					25: lambda: random.randrange(355, 377, 1),
					26: lambda: random.randrange(362, 387, 1),
					27: lambda: random.randrange(361, 383, 1),
					28: lambda: random.randrange(358, 384, 1),
					29: lambda: random.randrange(353, 374, 1),
					30: lambda: random.randrange(363, 383, 1),
				},
				3: {
					1: lambda: random.randrange(349, 357, 1),
					2: lambda: random.randrange(350, 359, 1),
					3: lambda: random.randrange(350, 358, 1),
					4: lambda: random.randrange(349, 354, 1),
					5: lambda: random.randrange(345, 362, 1),
					6: lambda: random.randrange(349, 361, 1),
					7: lambda: random.randrange(346, 357, 1),
					8: lambda: random.randrange(345, 358, 1),
					9: lambda: random.randrange(343, 359, 1),
					10: lambda: random.randrange(348, 356, 1),
					11: lambda: random.randrange(346, 358, 1),
					12: lambda: random.randrange(340, 356, 1),
					13: lambda: random.randrange(345, 354, 1),
					14: lambda: random.randrange(342, 359, 1),
					15: lambda: random.randrange(348, 357, 1),
					16: lambda: random.randrange(342, 356, 1),
					17: lambda: random.randrange(344, 359, 1),
					18: lambda: random.randrange(347, 357, 1),
					19: lambda: random.randrange(347, 354, 1),
					20: lambda: random.randrange(348, 358, 1),
					21: lambda: random.randrange(347, 357, 1),
					22: lambda: random.randrange(344, 362, 1),
					23: lambda: random.randrange(344, 357, 1),
					24: lambda: random.randrange(348, 360, 1),
					25: lambda: random.randrange(341, 360, 1),
					26: lambda: random.randrange(345, 363, 1),
					27: lambda: random.randrange(345, 354, 1),
					28: lambda: random.randrange(346, 358, 1),
					29: lambda: random.randrange(341, 374, 1),
					30: lambda: random.randrange(356, 377, 1),
				},
				4: {
					1: lambda: random.randrange(353, 380, 1),
					2: lambda: random.randrange(352, 370, 1),
					3: lambda: random.randrange(359, 380, 1),
					4: lambda: random.randrange(357, 378, 1),
					5: lambda: random.randrange(353, 373, 1),
					6: lambda: random.randrange(353, 375, 1),
					7: lambda: random.randrange(352, 379, 1),
					8: lambda: random.randrange(353, 373, 1),
					9: lambda: random.randrange(353, 371, 1),
					10: lambda: random.randrange(350, 369, 1),
					11: lambda: random.randrange(359, 382, 1),
					12: lambda: random.randrange(358, 377, 1),
					13: lambda: random.randrange(353, 378, 1),
					14: lambda: random.randrange(356, 378, 1),
					15: lambda: random.randrange(352, 373, 1),
					16: lambda: random.randrange(356, 388, 1),
					17: lambda: random.randrange(354, 375, 1),
					18: lambda: random.randrange(355, 366, 1),
					19: lambda: random.randrange(353, 513, 1),
					20: lambda: random.randrange(351, 368, 1),
					21: lambda: random.randrange(355, 375, 1),
					22: lambda: random.randrange(356, 493, 1),
					23: lambda: random.randrange(356, 376, 1),
					24: lambda: random.randrange(352, 374, 1),
					25: lambda: random.randrange(353, 501, 1),
					26: lambda: random.randrange(352, 378, 1),
					27: lambda: random.randrange(354, 374, 1),
					28: lambda: random.randrange(356, 496, 1),
					29: lambda: random.randrange(350, 370, 1),
					30: lambda: random.randrange(350, 375, 1),
				},
				5: {
					1: lambda: random.randrange(352, 496, 1),
					2: lambda: random.randrange(352, 378, 1),
					3: lambda: random.randrange(346, 381, 1),
					4: lambda: random.randrange(341, 416, 1),
					5: lambda: random.randrange(343, 356, 1),
					6: lambda: random.randrange(341, 357, 1),
					7: lambda: random.randrange(343, 375, 1),
					8: lambda: random.randrange(346, 355, 1),
					9: lambda: random.randrange(342, 487, 1),
					10: lambda: random.randrange(341, 362, 1),
					11: lambda: random.randrange(343, 359, 1),
					12: lambda: random.randrange(347, 361, 1),
					13: lambda: random.randrange(345, 359, 1),
					14: lambda: random.randrange(345, 359, 1),
					15: lambda: random.randrange(344, 487, 1),
					16: lambda: random.randrange(344, 356, 1),
					17: lambda: random.randrange(344, 355, 1),
					18: lambda: random.randrange(346, 357, 1),
					19: lambda: random.randrange(346, 359, 1),
					20: lambda: random.randrange(343, 369, 1),
					21: lambda: random.randrange(346, 498, 1),
					22: lambda: random.randrange(345, 358, 1),
					23: lambda: random.randrange(341, 356, 1),
					24: lambda: random.randrange(345, 368, 1),
					25: lambda: random.randrange(347, 354, 1),
					26: lambda: random.randrange(343, 485, 1),
					27: lambda: random.randrange(345, 367, 1),
					28: lambda: random.randrange(342, 358, 1),
					29: lambda: random.randrange(343, 356, 1),
					30: lambda: random.randrange(342, 356, 1),
				},
				6: {
					1: lambda: random.randrange(335, 347, 1),
					2: lambda: random.randrange(333, 401, 1),
					3: lambda: random.randrange(338, 348, 1),
					4: lambda: random.randrange(340, 350, 1),
					5: lambda: random.randrange(337, 355, 1),
					6: lambda: random.randrange(338, 352, 1),
					7: lambda: random.randrange(338, 349, 1),
					8: lambda: random.randrange(337, 375, 1),
					9: lambda: random.randrange(337, 350, 1),
					10: lambda: random.randrange(335, 351, 1),
					11: lambda: random.randrange(340, 402, 1),
					12: lambda: random.randrange(395, 572, 1),
					13: lambda: random.randrange(337, 390, 1),
					14: lambda: random.randrange(339, 347, 1),
					15: lambda: random.randrange(338, 346, 1),
					16: lambda: random.randrange(337, 351, 1),
					17: lambda: random.randrange(338, 351, 1),
					18: lambda: random.randrange(338, 350, 1),
					19: lambda: random.randrange(342, 349, 1),
					20: lambda: random.randrange(341, 347, 1),
					21: lambda: random.randrange(339, 349, 1),
					22: lambda: random.randrange(341, 350, 1),
					23: lambda: random.randrange(339, 345, 1),
					24: lambda: random.randrange(336, 349, 1),
					25: lambda: random.randrange(339, 348, 1),
					26: lambda: random.randrange(338, 350, 1),
					27: lambda: random.randrange(339, 347, 1),
					28: lambda: random.randrange(340, 348, 1),
					29: lambda: random.randrange(341, 354, 1),
					30: lambda: random.randrange(342, 349, 1),
				},
				7: {
					1: lambda: random.randrange(337, 349, 1),
					2: lambda: random.randrange(341, 347, 1),
					3: lambda: random.randrange(340, 348, 1),
					4: lambda: random.randrange(339, 350, 1),
					5: lambda: random.randrange(343, 350, 1),
					6: lambda: random.randrange(338, 351, 1),
					7: lambda: random.randrange(341, 351, 1),
					8: lambda: random.randrange(341, 348, 1),
					9: lambda: random.randrange(335, 350, 1),
					10: lambda: random.randrange(339, 348, 1),
					11: lambda: random.randrange(340, 350, 1),
					12: lambda: random.randrange(336, 349, 1),
					13: lambda: random.randrange(339, 348, 1),
					14: lambda: random.randrange(340, 347, 1),
					15: lambda: random.randrange(340, 349, 1),
					16: lambda: random.randrange(339, 351, 1),
					17: lambda: random.randrange(342, 353, 1),
					18: lambda: random.randrange(341, 350, 1),
					19: lambda: random.randrange(340, 349, 1),
					20: lambda: random.randrange(338, 352, 1),
					21: lambda: random.randrange(335, 351, 1),
					22: lambda: random.randrange(337, 346, 1),
					23: lambda: random.randrange(338, 350, 1),
					24: lambda: random.randrange(338, 347, 1),
					25: lambda: random.randrange(337, 353, 1),
					26: lambda: random.randrange(340, 356, 1),
					27: lambda: random.randrange(341, 349, 1),
					28: lambda: random.randrange(340, 349, 1),
					29: lambda: random.randrange(338, 350, 1),
					30: lambda: random.randrange(333, 347, 1),
				},
				8: {
					1: lambda: random.randrange(342, 348, 1),
					2: lambda: random.randrange(336, 350, 1),
					3: lambda: random.randrange(336, 350, 1),
					4: lambda: random.randrange(337, 346, 1),
					5: lambda: random.randrange(341, 348, 1),
					6: lambda: random.randrange(338, 349, 1),
					7: lambda: random.randrange(339, 348, 1),
					8: lambda: random.randrange(339, 352, 1),
					9: lambda: random.randrange(341, 348, 1),
					10: lambda: random.randrange(341, 347, 1),
					11: lambda: random.randrange(338, 351, 1),
					12: lambda: random.randrange(337, 351, 1),
					13: lambda: random.randrange(339, 350, 1),
					14: lambda: random.randrange(342, 348, 1),
					15: lambda: random.randrange(339, 348, 1),
					16: lambda: random.randrange(338, 345, 1),
					17: lambda: random.randrange(339, 346, 1),
					18: lambda: random.randrange(337, 350, 1),
					19: lambda: random.randrange(335, 348, 1),
					20: lambda: random.randrange(334, 354, 1),
					21: lambda: random.randrange(340, 351, 1),
					22: lambda: random.randrange(337, 354, 1),
					23: lambda: random.randrange(337, 347, 1),
					24: lambda: random.randrange(341, 348, 1),
					25: lambda: random.randrange(335, 347, 1),
					26: lambda: random.randrange(341, 348, 1),
					27: lambda: random.randrange(339, 352, 1),
					28: lambda: random.randrange(339, 350, 1),
					29: lambda: random.randrange(336, 348, 1),
					30: lambda: random.randrange(341, 350, 1),
				},
				9: {
					1: lambda: random.randrange(338, 348, 1),
					2: lambda: random.randrange(339, 348, 1),
					3: lambda: random.randrange(340, 351, 1),
					4: lambda: random.randrange(337, 348, 1),
					5: lambda: random.randrange(339, 347, 1),
					6: lambda: random.randrange(343, 349, 1),
					7: lambda: random.randrange(338, 351, 1),
					8: lambda: random.randrange(339, 353, 1),
					9: lambda: random.randrange(340, 347, 1),
					10: lambda: random.randrange(339, 351, 1),
					11: lambda: random.randrange(335, 349, 1),
					12: lambda: random.randrange(340, 347, 1),
					13: lambda: random.randrange(337, 351, 1),
					14: lambda: random.randrange(338, 352, 1),
					15: lambda: random.randrange(342, 348, 1),
					16: lambda: random.randrange(339, 348, 1),
					17: lambda: random.randrange(341, 349, 1),
					18: lambda: random.randrange(338, 348, 1),
					19: lambda: random.randrange(339, 349, 1),
					20: lambda: random.randrange(339, 350, 1),
					21: lambda: random.randrange(340, 345, 1),
					22: lambda: random.randrange(339, 347, 1),
					23: lambda: random.randrange(341, 353, 1),
					24: lambda: random.randrange(335, 348, 1),
					25: lambda: random.randrange(340, 351, 1),
					26: lambda: random.randrange(339, 349, 1),
					27: lambda: random.randrange(340, 348, 1),
					28: lambda: random.randrange(334, 347, 1),
					29: lambda: random.randrange(343, 348, 1),
					30: lambda: random.randrange(338, 348, 1),
				},
				10: {
					1: lambda: random.randrange(338, 351, 1),
					2: lambda: random.randrange(339, 351, 1),
					3: lambda: random.randrange(343, 349, 1),
					4: lambda: random.randrange(340, 349, 1),
					5: lambda: random.randrange(341, 348, 1),
					6: lambda: random.randrange(331, 351, 1),
					7: lambda: random.randrange(342, 349, 1),
					8: lambda: random.randrange(338, 347, 1),
					9: lambda: random.randrange(338, 347, 1),
					10: lambda: random.randrange(338, 347, 1),
					11: lambda: random.randrange(339, 352, 1),
					12: lambda: random.randrange(332, 349, 1),
					13: lambda: random.randrange(334, 346, 1),
					14: lambda: random.randrange(340, 347, 1),
					15: lambda: random.randrange(339, 349, 1),
					16: lambda: random.randrange(340, 351, 1),
					17: lambda: random.randrange(338, 354, 1),
					18: lambda: random.randrange(340, 348, 1),
					19: lambda: random.randrange(338, 347, 1),
					20: lambda: random.randrange(337, 351, 1),
					21: lambda: random.randrange(340, 354, 1),
					22: lambda: random.randrange(340, 349, 1),
					23: lambda: random.randrange(336, 347, 1),
					24: lambda: random.randrange(343, 351, 1),
					25: lambda: random.randrange(342, 349, 1),
					26: lambda: random.randrange(338, 349, 1),
					27: lambda: random.randrange(338, 346, 1),
					28: lambda: random.randrange(337, 347, 1),
					29: lambda: random.randrange(342, 351, 1),
					30: lambda: random.randrange(339, 349, 1),
				},
			},
			0.5: {
				
				1: {
					1: lambda: random.randrange(6653, 8417, 1),
					2: lambda: random.randrange(6606, 9032, 1),
					3: lambda: random.randrange(4921, 9097, 1),
					4: lambda: random.randrange(6356, 9517, 1),
					5: lambda: random.randrange(6998, 8498, 1),
					6: lambda: random.randrange(7126, 8970, 1),
					7: lambda: random.randrange(7278, 9181, 1),
					8: lambda: random.randrange(7451, 9336, 1),
					9: lambda: random.randrange(7456, 9477, 1),
					10: lambda: random.randrange(7938, 9564, 1),
				},
				2: {
					1: lambda: random.randrange(4406, 5608, 1),
					2: lambda: random.randrange(4257, 5476, 1),
					3: lambda: random.randrange(4623, 6317, 1),
					4: lambda: random.randrange(4269, 5106, 1),
					5: lambda: random.randrange(4350, 5344, 1),
					6: lambda: random.randrange(3252, 5471, 1),
					7: lambda: random.randrange(3858, 5672, 1),
					8: lambda: random.randrange(4417, 5869, 1),
					9: lambda: random.randrange(4555, 5803, 1),
					10: lambda: random.randrange(3236, 5517, 1),
				},
				3: {
					1: lambda: random.randrange(2745, 4216, 1),
					2: lambda: random.randrange(3481, 4579, 1),
					3: lambda: random.randrange(3314, 4313, 1),
					4: lambda: random.randrange(3058, 4290, 1),
					5: lambda: random.randrange(2877, 4122, 1),
					6: lambda: random.randrange(3617, 4501, 1),
					7: lambda: random.randrange(3686, 5289, 1),
					8: lambda: random.randrange(3595, 4620, 1),
					9: lambda: random.randrange(2665, 4710, 1),
					10: lambda: random.randrange(3528, 4606, 1),
				},
				4: {
					1: lambda: random.randrange(2902, 3565, 1),
					2: lambda: random.randrange(2832, 3499, 1),
					3: lambda: random.randrange(2691, 3607, 1),
					4: lambda: random.randrange(2844, 3643, 1),
					5: lambda: random.randrange(3093, 3520, 1),
					6: lambda: random.randrange(2747, 3406, 1),
					7: lambda: random.randrange(2892, 3669, 1),
					8: lambda: random.randrange(2905, 3470, 1),
					9: lambda: random.randrange(2865, 3662, 1),
					10: lambda: random.randrange(2958, 3603, 1),
				},
				5: {
					1: lambda: random.randrange(2644, 3110, 1),
					2: lambda: random.randrange(2700, 3239, 1),
					3: lambda: random.randrange(2651, 3264, 1),
					4: lambda: random.randrange(2719, 3400, 1),
					5: lambda: random.randrange(2664, 3122, 1),
					6: lambda: random.randrange(2299, 2699, 1),
					7: lambda: random.randrange(2345, 3442, 1),
					8: lambda: random.randrange(2629, 3072, 1),
					9: lambda: random.randrange(2755, 3166, 1),
					10: lambda: random.randrange(2125, 3069, 1),
				},
				6: {
					1: lambda: random.randrange(2125, 2945, 1),
					2: lambda: random.randrange(2478, 2987, 1),
					3: lambda: random.randrange(2479, 2815, 1),
					4: lambda: random.randrange(2468, 2850, 1),
					5: lambda: random.randrange(2349, 2834, 1),
					6: lambda: random.randrange(2331, 2952, 1),
					7: lambda: random.randrange(2251, 2846, 1),
					8: lambda: random.randrange(2463, 6740, 1),
					9: lambda: random.randrange(2536, 2920, 1),
					10: lambda: random.randrange(2412, 3104, 1),
				},
				7: {
					1: lambda: random.randrange(2195, 2587, 1),
					2: lambda: random.randrange(2284, 2580, 1),
					3: lambda: random.randrange(2270, 2502, 1),
					4: lambda: random.randrange(2259, 2617, 1),
					5: lambda: random.randrange(2119, 2703, 1),
					6: lambda: random.randrange(2276, 2638, 1),
					7: lambda: random.randrange(2321, 2688, 1),
					8: lambda: random.randrange(2285, 2610, 1),
					9: lambda: random.randrange(2139, 2574, 1),
					10: lambda: random.randrange(2283, 2593, 1),
				},
				8: {
					1: lambda: random.randrange(2194, 2457, 1),
					2: lambda: random.randrange(2191, 2487, 1),
					3: lambda: random.randrange(2000, 2530, 1),
					4: lambda: random.randrange(2024, 2407, 1),
					5: lambda: random.randrange(1986, 2393, 1),
					6: lambda: random.randrange(2221, 2382, 1),
					7: lambda: random.randrange(2150, 2442, 1),
					8: lambda: random.randrange(2186, 2354, 1),
					9: lambda: random.randrange(2186, 2402, 1),
					10: lambda: random.randrange(2079, 2395, 1),
				},
				9: {
					1: lambda: random.randrange(2084, 2276, 1),
					2: lambda: random.randrange(1875, 2215, 1),
					3: lambda: random.randrange(1998, 2191, 1),
					4: lambda: random.randrange(2081, 2247, 1),
					5: lambda: random.randrange(2099, 2287, 1),
					6: lambda: random.randrange(2097, 2317, 1),
					7: lambda: random.randrange(2000, 2238, 1),
					8: lambda: random.randrange(2025, 2288, 1),
					9: lambda: random.randrange(2113, 3034, 1),
					10: lambda: random.randrange(2079, 2331, 1),
				},
				10: {
					1: lambda: random.randrange(1960, 2137, 1),
					2: lambda: random.randrange(2013, 2129, 1),
					3: lambda: random.randrange(2006, 2151, 1),
					4: lambda: random.randrange(1986, 2194, 1),
					5: lambda: random.randrange(1842, 2196, 1),
					6: lambda: random.randrange(2015, 2151, 1),
					7: lambda: random.randrange(1895, 2182, 1),
					8: lambda: random.randrange(2029, 2164, 1),
					9: lambda: random.randrange(1926, 2178, 1),
					10: lambda: random.randrange(2058, 2188, 1),
				},
			}
		}
		return cs_degree[0.5][cs][cn]()
	
	@staticmethod
	def gen_cs_exec_time(cs):
		cs_degree = {
			0.1: {
				1: lambda: random.randrange(406.0, 447.0, 1),
				2: lambda: random.randrange(346.0, 386.0, 1),
				3: lambda: random.randrange(339.0, 349.0, 1),
				4: lambda: random.randrange(338.0, 350.0, 1),
				5: lambda: random.randrange(338.0, 351.0, 1),
				6: lambda: random.randrange(334.0, 347.0, 1),
				7: lambda: random.randrange(333.0, 348.0, 1),
				8: lambda: random.randrange(334.0, 348.0, 1),
				9: lambda: random.randrange(334.0, 352.0, 1),
				10: lambda: random.randrange(335.0, 350.0, 1),
			},
			0.2: {
				1: lambda: random.randrange(1798.0, 3391.0, 1),
				2: lambda: random.randrange(953.0, 1227.0, 1),
				3: lambda: random.randrange(830.0, 903.0, 1),
				4: lambda: random.randrange(708.0, 809.0, 1),
				5: lambda: random.randrange(722.0, 833.0, 1),
				6: lambda: random.randrange(703.0, 757.0, 1),
				7: lambda: random.randrange(697.0, 739.0, 1),
				8: lambda: random.randrange(687.0, 719.0, 1),
				9: lambda: random.randrange(687.0, 701.0, 1),
				10: lambda: random.randrange(678.0, 698.0, 1),
			},
			0.3: {
				1: lambda: random.randrange(6932.0, 9198.0, 1),
				2: lambda: random.randrange(2917.0, 3850.0, 1),
				3: lambda: random.randrange(2064.0, 2767.0, 1),
				4: lambda: random.randrange(1754.0, 2392.0, 1),
				5: lambda: random.randrange(1592.0, 2088.0, 1),
				6: lambda: random.randrange(1466.0, 1646.0, 1),
				7: lambda: random.randrange(1349.0, 1522.0, 1),
				8: lambda: random.randrange(1280.0, 1414.0, 1),
				9: lambda: random.randrange(1326.0, 1407.0, 1),
				10: lambda: random.randrange(1291.0, 1354.0, 1),
			},
			0.4: {
				1: lambda: random.randrange(7747.0, 9900.0, 1),
				2: lambda: random.randrange(3843.0, 4653.0, 1),
				3: lambda: random.randrange(2800.0, 3267.0, 1),
				4: lambda: random.randrange(2138.0, 2322.0, 1),
				5: lambda: random.randrange(1713.0, 2207.0, 1),
				6: lambda: random.randrange(1552.0, 1828.0, 1),
				7: lambda: random.randrange(1459.0, 1706.0, 1),
				8: lambda: random.randrange(1537.0, 1639.0, 1),
				9: lambda: random.randrange(1463.0, 1558.0, 1),
				10: lambda: random.randrange(1429.0, 1529.0, 1),
			},
			0.5: {
				1: lambda: random.randrange(10405.0, 13462.0, 1),
				2: lambda: random.randrange(5627.0, 7732.0, 1),
				3: lambda: random.randrange(3923.0, 5528.0, 1),
				4: lambda: random.randrange(3258.0, 4016.0, 1),
				5: lambda: random.randrange(2407.0, 3576.0, 1),
				6: lambda: random.randrange(2738.0, 3182.0, 1),
				7: lambda: random.randrange(2272.0, 2817.0, 1),
				8: lambda: random.randrange(2174.0, 2424.0, 1),
				9: lambda: random.randrange(2026.0, 2283.0, 1),
				10: lambda: random.randrange(1939.0, 2178.0, 1),
			},
		}
		return cs_degree[0.5][cs]()
	
	@staticmethod
	def gen_static_exec_time(cs, cn):
		"""
			Static Execution time data for examine reinforcement learning method converging
			to best execution time or not [Value functions]
		"""
		cs_degree = {
			0.1: {
				1: {
					1: 10000,
					2: 9900,
					3: 9800,
					4: 9700,
					5: 9600,
					6: 9500,
					7: 9400,
					8: 9300,
					9: 9200,
					10: 9100,
				},
				2: {
					1: 9000,
					2: 8900,
					3: 8800,
					4: 8700,
					5: 8600,
					6: 8500,
					7: 8400,
					8: 8300,
					9: 8200,
					10: 8100,
				},
				3: {
					1: 8000,
					2: 7900,
					3: 7800,
					4: 7700,
					5: 7600,
					6: 7500,
					7: 7400,
					8: 7300,
					9: 7200,
					10: 7100,
				},
				4: {
					1: 7000,
					2: 6900,
					3: 6800,
					4: 6700,
					5: 6600,
					6: 6500,
					7: 6400,
					8: 6300,
					9: 6200,
					10: 6100,
				},
				5: {
					1: 6000,
					2: 5900,
					3: 5800,
					4: 5700,
					5: 5600,
					6: 5500,
					7: 5400,
					8: 5300,
					9: 5200,
					10: 5100,
				},
				6: {
					1: 5000,
					2: 4900,
					3: 4800,
					4: 4700,
					5: 4600,
					6: 4500,
					7: 4400,
					8: 4300,
					9: 4200,
					10: 4100,
				},
				7: {
					1: 4000,
					2: 3900,
					3: 3800,
					4: 3700,
					5: 3600,
					6: 3500,
					7: 3400,
					8: 3300,
					9: 3200,
					10: 3100,
				},
				8: {
					1: 3000,
					2: 2900,
					3: 2800,
					4: 2700,
					5: 2600,
					6: 2500,
					7: 2400,
					8: 2300,
					9: 2200,
					10: 2100,
				},
				9: {
					1: 2000,
					2: 1900,
					3: 1800,
					4: 1700,
					5: 1600,
					6: 1500,
					7: 1400,
					8: 1300,
					9: 1200,
					10: 1100,
				},
				10: {
					1: 1000,
					2: 900,
					3: 800,
					4: 700,
					5: 600,
					6: 500,
					7: 400,
					8: 300,
					9: 300,
					10: 100,
				},
			}
		}
		return cs_degree[0.1][cs][cn]
	
	def write_record(self, cs, cn, filename="workflow_record.csv"):
		if not os.path.exists(os.getcwd() + "/{}".format(filename)):
			with open(os.getcwd() + "/{}".format(filename), 'w', newline='', encoding='utf-8') as r:
				# fieldnames = [
				# 	'center', 'degree', 'band', 'folder_name', 'cluster_size', 'cluster_num', 'exec_time', 'wall_time',
				# 	'cum_wall_time'
				# ]
				fieldnames = [
					'degrees', 'submit_dir', 'cluster_size', 'cluster_num'
				]
				writer = csv.DictWriter(r, fieldnames=fieldnames)
				writer.writeheader()
		
		with open(filename, 'a') as r:
			writer = csv.writer(r)
			writer.writerow([self.degrees, self.work_dir, cs, cn])
	
	def build(self, cluster_size=1, cluster_number=1):
		self.build_transformation_catalog(cluster_size, cluster_number)
		self.generate_region_hdr()
		self.process_color_band()
		self.write_rc()
		self.write_property_conf()
		self.pegasus_plan()


def main():
	a = Montage()
	a.build()
	a.pegasus_run()
	time.sleep(5)
	a.pegasus_remove()


if __name__ == "__main__":
	main()

from setuptools import setup, find_packages

setup(name='gym_workflow',
      packages=find_packages(),
      version='0.0.1',
      install_requires=['gym', 'astropy', 'karellen-sqlite'],
      dependecies_link=['https://github.com/pegasus-isi/montage-workflow-v2.git'],
      include_package_data=True,
      )

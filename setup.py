from setuptools import setup

setup(name='gym_workflow',
      author='Rayson Leong',
      author_email='rayson@um.edu.my',
      url='https://github.com/rayson1223/gym-workflow.git',
      version='0.1',
      install_requires=['gym', 'astropy', 'pandas', 'numpy', 'matplotlib', 'PyGnuplot'],
      dependency_links=['git+https://github.com/pegasus-isi/montage-workflow-v2.git#egg=montage-gen-2.0',
                        'git+https://github.com/Caltech-IPAC/Montage.git#egg=5.0',
                        ],
      )

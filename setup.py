from setuptools import setup, find_packages

setup(name="beyondCV",
      version = "0.1",
      packages = find_packages(),
      description = "Beyond cosmic variance",
      url = "https://github.com/thibautlouis/beyondCV",
      author = "Thibault Louis",
      author_email = "thibault.louis@lal.in2p3.fr",
      keywords = ["CMB", "cosmic variance", "planck", "SO"],
      classifiers = ["Intended Audience :: Science/Research",
                     "Topic :: Scientific/Engineering :: Astronomy",
                     "Operating System :: OS Independent",
                     "Programming Language :: Python :: 2.7",
                     "Programming Language :: Python :: 3.7"],
      install_requires = ["pyyaml"],
      entry_points = {
        "console_scripts": ["beyondCV=beyondCV.beyondCV:main"],
      }
)

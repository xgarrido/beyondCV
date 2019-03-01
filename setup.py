from setuptools import setup, find_packages
from beyondCV import __author__, __version__, __url__

setup(name="beyondCV",
      version = __version__,
      packages = find_packages(),
      description = "Beyond cosmic variance",
      url = __url__,
      author = __author__,
      keywords = ["CMB", "cosmic variance", "planck", "SO"],
      classifiers = ["Intended Audience :: Science/Research",
                     "Topic :: Scientific/Engineering :: Astronomy",
                     "Operating System :: OS Independent",
                     "Programming Language :: Python :: 2.7",
                     "Programming Language :: Python :: 3.7"],
      install_requires = ["camb", "cobaya"],
      entry_points = {
        "console_scripts": ["beyondCV = beyondCV.beyondCV:main"],
      }
)

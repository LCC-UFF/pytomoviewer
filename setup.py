from setuptools import setup, Command, find_packages
import os


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system('rm -vrf build dist *.pyc *.tgz python/*.egg-info')

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pytomoviewer",
    version="0.0.1",
    author="LCC-IC-UFF",
    maintainer_email="andremaues@id.uff.br",
    description="Tool for the visualization of micro-computed tomography images.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LCC-UFF/pytomoviewer",
    platforms=["Linux", "Mac", "Windows"],
    packages=find_packages(),
    cmdclass={'clean': CleanCommand},
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "PyQt5==5.12",
        "scikit-image",
    ],
)

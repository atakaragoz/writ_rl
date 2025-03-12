from setuptools import find_packages, setup

# read the contents of requirements.txt
with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="writ_tools",
    packages=find_packages(),
    install_requires=required,  # pass the list of requirements here
)

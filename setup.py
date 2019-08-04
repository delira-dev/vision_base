from setuptools import setup, find_packages

import os
import re


def resolve_requirements(file):
    requirements = []
    with open(file) as f:
        req = f.read().splitlines()
        for r in req:
            if r.startswith("-r"):
                requirements += resolve_requirements(
                    os.path.join(os.path.dirname(file), r.split(" ")[1]))
            else:
                requirements.append(r)
    return requirements


def read_file(file):
    with open(file) as f:
        content = f.read()
    return content


requirements = resolve_requirements(os.path.join(os.path.dirname(__file__),
                                                 'requirements.txt'))

readme = read_file(os.path.join(os.path.dirname(__file__), "README.md"))
license = read_file(os.path.join(os.path.dirname(__file__), "LICENSE"))

setup(
    name='deliravision-base',
    packages=find_packages(),
    url='https://github.com/delira-dev/vision',
    test_suite="unittest",
    long_description=readme,
    long_description_content_type='text/markdown',
    license=license,
    install_requires=requirements,
    tests_require=["coverage"],
    python_requires=">=3.5",
    maintainer="Michael Baumgartner",
    maintainer_email="michael.baumgartner@rwth-aachen.de",
)

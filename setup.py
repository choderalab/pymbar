from setuptools import setup, find_packages
import versioneer


long_description = """
Pymbar (https://simtk.org/home/pymbar) is a library
that provides tools for optimally combining simulations
from multiple thermodynamic states using maximum likelihood
methods to compute free energies (normalization constants)
and expectation values from all of the samples simultaneously.
"""

setup(
    name="pymbar",
    author="Levi N. Naden and Jaime Rodriguez-Guerra and Michael R. Shirts and John D. Chodera",
    author_email="levi.naden@choderalab.org, jaime.rodriguez-guerra@choderalab.org, michael.shirts@virginia.edu, john.chodera@choderalab.org",
    description="Python implementation of the multistate Bennett acceptance ratio (MBAR) method",
    license="MIT",
    keywords="molecular mechanics, forcefield, Bayesian parameterization",
    url="http://github.com/choderalab/pymbar",
    packages=find_packages(),
    long_description=long_description[1:],
    classifiers=["License :: OSI Approved :: MIT License", "Programming Language :: Python :: 3"],
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    python_requires=">=3.6",
    install_requires=["numpy>=1.12",
                      "scipy",
                      "numexpr",
                      "jaxlib;platform_system!='Windows'",
                      "jax;platform_system!='Windows'"
                      ],
)

from setuptools import setup, find_packages

setup(
    name="greedy-dynamic-quadruped-gait",
    version="0.1.0",
    description="Greedy Dynamic Quadruped Gait",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "rsl-rl @ git+https://github.com/leggedrobotics/rsl_rl.git@v1.0.2"
    ],
)
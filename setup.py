from setuptools import setup, find_packages

setup(
    name="gaitnet",
    version="0.1.0",
    description="Greedy Dynamic Quadruped Gait",
    packages=find_packages(),
    package_dir={"": "."},
    install_requires=[
        "torch",
        "rsl-rl @ git+https://github.com/leggedrobotics/rsl_rl.git@v1.0.2"
    ],
)
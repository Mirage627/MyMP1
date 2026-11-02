"""Editable source install for the preserved EfficientFlow import namespace."""
from setuptools import find_namespace_packages, setup

setup(
    name="MyMP1",
    version="0.1.0",
    description="MP1, MGAD and H second-moment controlled experiments",
    packages=["EfficientFlow"] + [
        "EfficientFlow." + name
        for name in find_namespace_packages(
            where="EfficientFlow",
            include=["codecs*", "common*", "dataset*", "env*", "experiments*",
                     "gym_util*", "model*", "models*", "policy*", "shared_memory*", "workspace*"],
        )
    ],
    package_data={"EfficientFlow": ["config/*.yaml", "config/task/*.yaml"]},
    python_requires=">=3.9",
)

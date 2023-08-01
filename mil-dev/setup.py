import os
import setuptools


path = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(path, "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mil",
    version="0.0.1",
    author="Lee Cooper",
    author_email="lee.cooper@northwestern.edu",
    description="A TensorFlow 2 package for pathology imaging",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/PathologyDataScience/mil",
    packages=setuptools.find_packages(exclude=["tests"]),
    package_dir={
        "mil": "mil",
    },
    install_requires=[
        "numpy",
        "pandas",
        "tensorflow>=2.6",
    ],
    extras_require={
        "ray": ["ray[tune]"],
    },
    license="Apache Software License 2.0",
    include_package_data=True,
    keywords="mil",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    zip_safe=False,
    python_requires=">=3.6",
)

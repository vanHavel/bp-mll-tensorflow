from setuptools import setup

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

setup(
    name='bpmll',
    version='1.0.0',
    description='BP-MLL loss function for tensorflow',
    keywords="bpmll",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/vanHavel/bp-mll-tensorflow',
    author='Lukas Huwald',
    author_email='dev.lukas.huwald@gmail.com',
    license='MIT',
    packages=['bpmll'],
    zip_safe=True,
    install_requires=[
        "tensorflow"
    ],
    python_requires='>=3.5',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ]
)
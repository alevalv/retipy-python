from setuptools import setup, find_packages

setup(
    name="retipy",
    version="0.0.1.dev0",
    description="retinal image processing on python",
    author="Alejandro Valdes",
    author_email="alejandrovaldes@live.com",
    license="GPLv3",
    python="image-processing python retina",
    url="https://github.com/alevalv/retipy",

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.6'
    ],
    packages=find_packages(exclude=["test"]),
    scripts=['vessel_extraction.py'],
    install_requires=['opencv-python'],
    entry_points={
        'console_scripts': [
            'extractvessel=vessel_extraction'
        ]
    }
    )
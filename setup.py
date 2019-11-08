from distutils.core import setup
from setuptools import find_packages


def setup_package():
    config = {
        'name': 'handwriting-generation',
        'version': '0.0.1',
        'description': 'handwriting generation',
        'author': 'TC',
        'license': 'MIT',
        'tests_require': ['pytest'],
        'packages': find_packages(
            exclude=("tests", )),
        'keywords': [
            'handwriting generation',
        ]
    }

    setup(**config)


if __name__ == '__main__':
    setup_package()

import re

import pkg_resources as pkg

from pathlib import Path
from setuptools import setup, find_packages

FILE = Path(__file__).resolve()
PARENT = FILE.parent
README = (PARENT / 'README.md').read_text(encoding='utf-8')
REQUIREMENTS = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements(
    (PARENT / 'requirements.txt').read_text())]

def get_version():
    file = PARENT / 'airgym/__init__.py'
    return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', file.read_text(encoding='utf-8'), re.M)[1]

setup(
    name='airgym',
    version=get_version(),
    python_requires='>=3.6',
    packages=find_packages(),
    include_package_data=True,
    install_requires=REQUIREMENTS,
    entry_points={
        'console_scripts': [
            'airgym=airgym.cli:main',
        ]
    },
    author='SAUVECANNE, Baptiste',
    author_email='contact@baptistesauvecanne.fr',
    description='OpenAI Gym environment for training reinforcement learning agents on an XPlane simulator',
    long_description=README,
    long_description_content_type='text/markdown',
    license='GPL-3.0',
    url='https://github.com/zacjiker/airgym',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords=["openai", "gym", "xplane", "reinforcement learning", "rl",
              "environment", "ai", "artificial intelligence", "machine learning", "ml"]
)

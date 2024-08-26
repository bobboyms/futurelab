from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='futurelabs',
    version='0.1.0',
    description='FutureLabs: An awesome model log em chart',
    author='Thiago Luiz Rodrigues',
    author_email='thiago@rodriguesthiago.me',
    url='https://github.com/bobboyms/futurelab',
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'futurelabs=futurelabs.main:main',
        ],
    },
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

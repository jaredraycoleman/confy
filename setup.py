from setuptools import setup

setup(
    name='confy',
    author="Jared Coleman",
    author_email="jaredraycoleman@gmail.com",
    version='0.0.1',
    packages=['confy'],
    include_package_data=True,
    install_requires=[
        "pandas",
        "thefuzz[speedup]",
        "html2text",
        "requests",
        "beautifulsoup4",
    ],
    entry_points={
        'console_scripts': [
            'confy = confy:main',
        ],
    }
)
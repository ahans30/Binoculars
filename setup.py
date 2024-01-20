from setuptools import setup, find_packages

setup(
    name='Binoculars',
    version='0.0.10',
    packages=find_packages(),
    url='https://github.com/ahans30/Binoculars',
    license=open("LICENSE.md", "r", encoding="utf-8").read(),
    author='Authors of "Binoculars: Zero-Shot Detection of LLM-Generated Text"',
    author_email='ahans1@umd.edu',
    description='A language model generated text detector.',
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    install_requires=open("requirements.txt", "r", encoding="utf-8").read().splitlines(),
)

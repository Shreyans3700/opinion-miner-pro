from setuptools import setup, find_packages


# Read requirements from requirements.txt
def get_requirements(file_path: str):
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")

    return requirements


setup(
    name="review_intelligence_system",
    version="0.1.0",
    author="Shreyansh Pandey",
    author_email="pandeyshreyansh46@gmail.com",
    description="Aspect-based sentiment analysis system for extracting insights from reviews",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

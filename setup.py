from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


hyphen_e_dot = "-e ."

def get_requirements(file_path:str)->list[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        if  hyphen_e_dot in requirements:
            requirements.remove(hyphen_e_dot)


    return requirements


__version__ = "0.0.0"

REPO_NAME = "Hate-Speech-Classification"
AUTHOR_USER_NAME = "santosh3110"
SRC_REPO = "hate_speech_classifier"
AUTHOR_EMAIL = "santoshkumarguntupalli@gmail.com"
LIST_OF_REQUIREMENTS = get_requirements("requirements.txt")


setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="Hate Speech Classification using BiLSTM + Conv1D + GloVe",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    packages=find_packages(),
    license="Apache License 2.0",
    python_requires=">=3.7",
    install_requires=LIST_OF_REQUIREMENTS
)

import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    desc = f.read()


setuptools.setup(
    name="gtml-baseline",
    version="0.0.1",
    author="SJTU-MARL",
    description="A baseline satisfies SJTU course AI3617",
    long_description=desc,
    long_description_content_type="text/markdown",
    url="https://github.com/sjtu-marl/GTML22_baseline",
    project_urls={
        "Bug Tracker": "https://github.com/sjtu-marl/GTML22_baseline/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "ray==1.12.0",
        "pettingzoo==1.17.0",
        "numba==0.55.1",
        "colorlog==6.6.0",
        "torch==1.11.0",
    ],
)

from setuptools import setup

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="dapr-agents",
    version="0.11.2",
    description="Agentic Workflows Made Simple",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
import yaml
from jinja2 import Template
from langsmith import Client
from os import path

ls_client = Client()
"""
The LangSmith client used to interact with the prompt registry.
"""


def prompt_template_config(prompt_key: str) -> Template:
    """
    Fetches a prompt template from a YAML configuration file, starting from a prompt key.
    It assumes that the YAML file is named `{prompt_key}.yaml`.
    """

    filepath: str = path.join(path.dirname(__file__), "files", f"{prompt_key}.yaml")

    with open(filepath, "r") as file:
        config = yaml.safe_load(file)

    template_content = config["prompts"][prompt_key]

    template = Template(template_content)

    return template


def prompt_template_registry(prompt_name: str) -> Template:
    """
    Fetches a prompt template from the LangSmith prompt registry.
    """

    template_content = ls_client.pull_prompt(prompt_name).messages[0].prompt.template

    template = Template(template_content)

    return template

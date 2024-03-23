import os
import pathlib

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_community.llms.ollama import Ollama
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

WORK_DIR = pathlib.Path(__file__).parent.parent.resolve()
PROMPTS_DIR = WORK_DIR / 'prompts'


class Model:
    def __init__(self, model_type):
        self.model = Ollama(model=model_type)
        self.response_schemas = [
            ResponseSchema(name="newComment", description="the new comment"),
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(self.response_schemas)
        self.format_instructions = self.output_parser.get_format_instructions()

        with open(PROMPTS_DIR / 'v1.txt', 'r') as f:
            template = os.linesep.join(f.readlines())

        self.prompt = PromptTemplate(
            template=template,
            input_variables=["old_method", "old_comment", "new_method"],
            partial_variables={"format_instructions": ''},
        )
        self.chain = self.prompt | self.model | StrOutputParser()

    def resolve(self, old_method, new_method, old_comment):
        while True:
            try:
                return self.chain.invoke({
                    'old_method': old_method,
                    'new_method': new_method,
                    'old_comment': old_comment
                })
            except OutputParserException as e:
                print(e)

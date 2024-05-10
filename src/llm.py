import os
import pathlib

import ollama
from langchain_community.llms.ollama import Ollama
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

WORK_DIR = pathlib.Path(__file__).parent.parent.resolve()
PROMPTS_DIR = WORK_DIR / 'prompts'


class Model:
    def __init__(self, model_type):
        with open(PROMPTS_DIR / 'v2.txt', 'r') as f:
            self.template = os.linesep.join(f.readlines())
        self.model_type = model_type

    def resolve(self, old_method, new_method, old_comment):
        pass


class LangChainModel(Model):
    def __init__(self, model_type):
        super().__init__(model_type)

        self.model = Ollama(model=model_type)
        self.prompt = PromptTemplate(
            template=self.template,
            input_variables=["old_method", "old_comment", "new_method"],
        )

        self.retriever = None
        self.chain = self.prompt | self.model | StrOutputParser()

    def resolve(self, old_method, new_method, old_comment):
        while True:
            try:
                args = {
                    'old_method': old_method,
                    'new_method': new_method,
                    'old_comment': old_comment
                }
                if self.retriever is not None:
                    args['context'] = self.retriever

                return self.chain.invoke(args)
            except OutputParserException as e:
                print(e)


class OllamaModel(Model):
    def __init__(self, model_type, rag_src=None):
        super().__init__(model_type)

    def resolve(self, old_method, new_method, old_comment):
        prompt = self.template.format(
            old_comment=old_comment,
            old_method=old_method,
            new_method=new_method,
        )

        result = ollama.generate(
            model=self.model_type,
            prompt=prompt,
            stream=False
        )

        return result['response']

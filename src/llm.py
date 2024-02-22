from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_community.llms.ollama import Ollama
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


class Model:
    def __init__(self, model_type):
        self.model = Ollama(model=model_type)
        self.response_schemas = [
            ResponseSchema(name="newComment", description="the new comment"),
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(self.response_schemas)
        self.format_instructions = self.output_parser.get_format_instructions()
        self.prompt = PromptTemplate(
            template=
            '''
            Read the following Java method: ```{old_method}```
            Read the following javadoc comment belonging to the method mentioned before: {old_comment}
            Then the Java method was modified to ```{new_method}```
            Please change the comment to fit the new method. 
            The fewer changes, the better. Answer the comment only.
            {format_instructions}''',
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

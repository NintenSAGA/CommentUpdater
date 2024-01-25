import json

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import StrOutputParser


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


myModel = Model('mistral-openorca')
with open('../data/raw/test.jsonl', 'r') as f:
    lines = f.readlines()
    for nr_line in range(0, 2):
        raw_json = lines[nr_line]
        print(f"## Case {nr_line + 1}")
        parsed = json.loads(raw_json)
        _old_method = parsed['src_method']
        _new_method = parsed['dst_method']
        _old_comment = parsed['src_javadoc']
        result = myModel.resolve(_old_method, _new_method, _old_comment)
        # print(f"Old     : {_old_comment}")
        # print(f"Expected: {parsed['dst_javadoc']}")
        # print(f'Actual  : {result.strip()}')
        # print()
        output_dict = {
            'sample_id': parsed['sample_id'],
            'full_name': parsed['full_name'],
            'commit_id': parsed['commit_id'],
            'src_method': _old_method,
            'dst_method': _new_method,
            'src_javadoc': _old_comment,
            'dst_javadoc': parsed['dst_javadoc'],
            'act_javadoc': result.strip()
        }
        output_s = json.dumps(output_dict, indent=2)
        print(output_s)

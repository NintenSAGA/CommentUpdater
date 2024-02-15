import datetime
import json

import jsonlines
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

NUM = 100
nr_line = 0

result_file_name = f'result-{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl'
with jsonlines.open('../data/raw/test.jsonl') as reader:
    with jsonlines.open(f'../data/results/{result_file_name}', mode='w') as writer:
        for parsed in reader:
            if nr_line == NUM:
                break
            print(f"## Case {nr_line + 1}")
            nr_line += 1
            _old_method = parsed['src_method']
            _new_method = parsed['dst_method']
            _old_comment = parsed['src_javadoc']
            result = myModel.resolve(_old_method, _new_method, _old_comment)
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
            # print(output_s)
            writer.write(output_dict)

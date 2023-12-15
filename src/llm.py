from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

INFER_COMMENT_TEMPLATE_1 = '''
Old Method: ```{old_method}```

New Method: ```{new_method}```

Old Comment: ```{old_comment}```

Please infer the new comment according to the code modification.
Rules: No explanation. No natural words. Perform the minimum modification Avoid any lose of information
'''


class LLM(object):

    def __init__(self, model: str):
        self.llm = Ollama(model=model)

    def infer_new_comment(self, old_method: str, new_method: str, old_comment: str) -> str:
        prompt = ChatPromptTemplate.from_template(INFER_COMMENT_TEMPLATE_1)
        output_parser = StrOutputParser()

        chain = prompt | self.llm | output_parser

        return chain.invoke({
            'new_method': new_method,
            'old_method': old_method,
            'old_comment': old_comment,
        })

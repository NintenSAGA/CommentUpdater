from langchain.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

llm = Ollama(
    model="llama2"
)

old_method = ('private String getSessionFileName(String sessionIdentifier , boolean createSessionFolder) {File '
              'sessionFolder = folders.get(sessionIdentifier , createSessionFolder); return new File(sessionFolder , '
              '" data" ). getAbsolutePath (); }')
new_method = ('private String getSessionFileName(String sessionIdentifier) {File sessionFolder = folders.get('
              'sessionIdentifier , true); return new File(sessionFolder , " data" ). getAbsolutePath (); }')
old_comment = ('If the session folder (folder that contains the ﬁle) does not exist and createSessionFolder is true, '
               'the folder will be created.')

source = '''
Old Method: ```{old_method}```

New Method: ```{new_method}```

Old Comment: ```{old_comment}```

Please infer the new comment according to the code modification.
Rules: No explanation. No natural words. Perform the minimum modification Avoid any lose of information
'''

prompt = ChatPromptTemplate.from_template(source)
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

answer = chain.invoke({
    'new_method': new_method,
    'old_method': old_method,
    'old_comment': old_comment,
})

print(answer)

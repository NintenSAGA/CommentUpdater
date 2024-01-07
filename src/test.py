from operator import itemgetter

from langchain.schema import StrOutputParser
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate

prompt1 = ChatPromptTemplate.from_template("what is the city {person} is from?")
prompt2 = ChatPromptTemplate.from_template(
    "what country is the city {city} in? respond in {language}"
)

model = Ollama(model='llama2')

chain1 = prompt1 | model | StrOutputParser()

chain2 = (
    {"city": chain1, "language": itemgetter("language")}
    | prompt2
    | model
    | StrOutputParser()
)

answer = chain2.invoke({"person": "obama", "language": "spanish"})

print('answer: ' + answer)
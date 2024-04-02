import os
import pathlib

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

WORK_DIR = pathlib.Path(__file__).parent.parent.resolve()
PROMPTS_DIR = WORK_DIR / 'prompts'


class Model:
    def __init__(self, model_type, rag_src=None):
        self.model = Ollama(model=model_type)
        self.response_schemas = [
            ResponseSchema(name="newComment", description="the new comment"),
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(self.response_schemas)
        self.format_instructions = self.output_parser.get_format_instructions()

        with open(PROMPTS_DIR / 'v2.txt', 'r') as f:
            template = os.linesep.join(f.readlines())

        self.prompt = PromptTemplate(
            template=template,
            input_variables=["old_method", "old_comment", "new_method"],
            partial_variables={"format_instructions": ''},
        )

        if rag_src is None:
            self.retriever = None
            self.chain = self.prompt | self.model | StrOutputParser()
        else:
            loader = JSONLoader(
                file_path=rag_src,
                jq_schema='.',
                text_content=False,
                json_lines=True)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
            splits = text_splitter.split_documents(docs)
            vectorstore = Chroma.from_documents(documents=splits, embedding=OllamaEmbeddings(model=model_type))
            self.retriever = vectorstore.as_retriever()
            self.prompt += '\nContext: {context}'
            self.chain = (
                self.prompt | self.model | StrOutputParser()
            )

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

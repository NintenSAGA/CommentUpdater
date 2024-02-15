import datetime
import difflib
import json

import jsonlines
import rouge
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import StrOutputParser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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


def calculate_diff(original: str, updated: str):
    matcher = difflib.SequenceMatcher(lambda x: x == " ", original, updated)
    print(matcher.get_opcodes())
    modifications = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            # Add the modification pair (old, new) to the list
            modifications.append((original[i1:i2], updated[j1:j2]))

    return modifications


def cal_cosine_similarity(text1, text2):
    # Create a TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Transform texts to TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    # Calculate Cosine Similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    return similarity


def calc_and_filter(candidates, dst_method, src_javadoc, m, p):
    _rouge = rouge.Rouge()
    scores = _rouge.get_scores(hyps=candidates,
                               refs=[src_javadoc] * len(candidates))
    cand_tuples = []
    for _i, score in enumerate(scores):
        hyp = candidates[_i]
        recall = score['rouge-l']['r']
        cs = cal_cosine_similarity(dst_method, hyp)
        cand_tuples.append((hyp, recall, cs))

    cand_tuples = sorted(cand_tuples, key=lambda x: x[2], reverse=True)
    cand_tuples = list(filter(lambda x: x[0] != src_javadoc, cand_tuples))
    cand_tuples = cand_tuples[:min(m, len(cand_tuples))]
    cand_tuples = sorted(cand_tuples, key=lambda x: x[1], reverse=True)
    cand_tuples = cand_tuples[:min(len(cand_tuples), p)]

    for t in cand_tuples:
        print(f'recall: {t[1]:.2f} cs: {t[2]:.2f} - {t[0]}')

    return list(map(lambda x: x[0], cand_tuples))


myModel = Model('mistral-openorca')

NUM = 5
nr_line = 0

selected = {5062276, 2336970, 3992375, 440280}

result_file_name = f'result-{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl'
with jsonlines.open('../data/raw/test.jsonl') as reader:
    with jsonlines.open(f'../data/results/{result_file_name}', mode='w') as writer:
        for parsed in reader:
            _sample_id = parsed['sample_id']
            if len(selected) != 0:
                if _sample_id not in selected:
                    continue
            elif nr_line == NUM:
                break
            print(f"## Case {nr_line + 1} (ID: {_sample_id})")
            nr_line += 1
            _old_method = parsed['src_method']
            _new_method = parsed['dst_method']
            _old_comment = parsed['src_javadoc']
            _exp_comment = parsed['dst_javadoc']

            print(f'Expected: {_exp_comment}')
            _n = 10
            _m = 5
            _p = 3
            _candidates = []
            for i in range(_n):
                result = myModel.resolve(_old_method, _new_method, _old_comment)
                result = result.rstrip('<|im_end|>')
                _candidates.append(result)
            _n_candidates = calc_and_filter(_candidates, _new_method, _old_comment, _m, _p)

            # result = myModel.resolve(_old_method, _new_method, _old_comment)
            output_dict = {
                'sample_id': parsed['sample_id'],
                'full_name': parsed['full_name'],
                'commit_id': parsed['commit_id'],
                'src_method': _old_method,
                'dst_method': _new_method,
                'src_javadoc': _old_comment,
                'dst_javadoc': _exp_comment,
                # 'act_javadoc': result.strip()
                'act_javadoc': _n_candidates
            }
            output_s = json.dumps(output_dict, indent=2)
            # print(output_s)
            writer.write(output_dict)

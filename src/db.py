import sqlite3

import jsonlines

import eval
import main
import sorter

DB_PATH = main.WORK_DIR / 'result' / 'result.db'

con = sqlite3.connect(DB_PATH)

cur = con.cursor()

cur.execute('DROP TABLE IF EXISTS TestData')
cur.execute('''
CREATE TABLE TestData(id, sampleId, fullname, commitId, oldCode, newCode, origin, reference)
''')

cur.execute('DROP TABLE IF EXISTS TestResult')
cur.execute('''
CREATE TABLE TestResult(approach, id, hypothesis, ed, red, gleu, meteor, rouge_recall, rouge_f1)
''')

data1_ = []
with jsonlines.open('../data/raw/test_clean.jsonl') as reader:
    id_cnt_ = 0
    for obj in reader:
        data1_.append(
            (id_cnt_, obj['sample_id'], obj['full_name'], obj['commit_id'], obj['src_method'], obj['dst_method'],
             obj['src_desc'], obj['dst_desc'])
        )
        id_cnt_ += 1
cur.executemany('INSERT INTO TestData VALUES(?, ?, ?, ?, ?, ?, ?, ?)', data1_)
con.commit()

llm_result_ = eval.evaluate('../result/candidates/candidates-20240407_154640.jsonl')  # openorca
llm_top_result_ = sorter.get_first_candidates(llm_result_, lambda x: sorter.sort_by_evaluation_metric(x, 'gleu', True))
cup_result = eval.evaluate('../result/baseline/CUP.jsonl', 'CUP')
hebcup_result = eval.evaluate('../result/baseline/HebCup.jsonl', 'HebCup')


def insert_result(approach, array):
    id_cnt = 0
    data1 = []
    for e in array:
        data1.append((
            approach, id_cnt, e['content'],
            e['ed'], e['red'], e['gleu'], e['meteor'], e['rouge-recall'], e['rouge-f1'],
        ))
        id_cnt += 1
    cur.executemany('INSERT INTO TestResult VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)', data1)
    con.commit()


insert_result('LLM', llm_top_result_)
insert_result('CUP', cup_result)
insert_result('HebCup', hebcup_result)

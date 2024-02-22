import datetime
import json
import pathlib

import yaml

import jsonlines

from algo import calc_and_filter
from llm import Model

WORK_DIR = pathlib.Path(__file__).parent.parent.resolve()
CONFIG_DIR = WORK_DIR / 'config'

if __name__ == '__main__':
    with open((CONFIG_DIR / 'config.yml'), 'r') as file:
        config = yaml.safe_load(file)
    test_data = config['testData']

    myModel = Model('mistral-openorca')

    NUM = test_data['num']
    SELECTED = test_data['selected']
    PATH = test_data['path']

    nr_line = 0

    result_file_name = f'result-{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl'
    with jsonlines.open(PATH) as reader:
        with jsonlines.open(f'../data/results/{result_file_name}', mode='w') as writer:
            for parsed in reader:
                _sample_id = parsed['sample_id']
                if SELECTED is not None and len(SELECTED) != 0:
                    if _sample_id not in SELECTED:
                        continue
                elif nr_line == NUM:
                    break
                print(f"## Case {nr_line + 1} (ID: {_sample_id})")
                nr_line += 1
                _old_method = parsed['src_method']
                _new_method = parsed['dst_method']
                _old_comment = parsed['src_desc']
                _exp_comment = parsed['dst_desc']

                print(f'Expected: {_exp_comment}')
                _n = 10
                _m = 5
                _p = 3
                _candidates = set()
                for i in range(_n):
                    result = myModel.resolve(_old_method, _new_method, _old_comment)
                    result = result.rstrip('<|im_end|>')
                    _candidates.add(result)
                _n_candidates = calc_and_filter(list(_candidates), _new_method, _old_comment, _m, _p)

                # result = myModel.resolve(_old_method, _new_method, _old_comment)
                output_dict = {
                    'sample_id': parsed['sample_id'],
                    'full_name': parsed['full_name'],
                    'commit_id': parsed['commit_id'],
                    'src_method': _old_method,
                    'dst_method': _new_method,
                    'src_desc': _old_comment,
                    'dst_desc': _exp_comment,
                    # 'act_javadoc': result.strip()
                    'act_desc': _n_candidates
                }
                output_s = json.dumps(output_dict, indent=2)
                # print(output_s)
                writer.write(output_dict)

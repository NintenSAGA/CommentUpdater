import datetime
import json
import pathlib

import jsonlines
import yaml

from algo import calc_and_filter
from llm import Model
from dotenv import load_dotenv

WORK_DIR = pathlib.Path(__file__).parent.parent.resolve()
CONFIG_DIR = WORK_DIR / 'config'

load_dotenv()

if __name__ == '__main__':
    with open((CONFIG_DIR / 'config.yml'), 'r') as file:
        config = yaml.safe_load(file)
    test_data = config['testData']
    params = config['params']

    # myModel = Model('mistral-openorca')
    model_name = config['model']
    myModel = Model(model_name)
    print(f'Model: {model_name}')

    NUM = test_data['num']
    SELECTED = test_data['selected']
    PATH = test_data['path']
    PATH = WORK_DIR / PATH

    nr_line = 0

    result_file_name = f'result-{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl'
    with jsonlines.open(PATH) as reader:
        # with jsonlines.open(f'../data/results/{result_file_name}', mode='w') as writer:
        for parsed in reader:
            _sample_id = parsed['sample_id']
            if SELECTED is not None and len(SELECTED) != 0:
                if _sample_id not in SELECTED:
                    continue
            elif nr_line == NUM:
                break

            print(f"\n## Case {nr_line + 1} (ID: {_sample_id})")

            nr_line += 1
            _old_method = parsed['src_method']
            _new_method = parsed['dst_method']
            _old_comment = parsed['src_javadoc']
            _exp_comment = parsed['dst_javadoc']

            print(f'1️⃣Original: {_old_comment}')
            print(f'1️⃣Expected: {_exp_comment}')

            _n = params['nr_gen']
            _candidates = set()
            for i in range(_n):
                result = myModel.resolve(_old_method, _new_method, _old_comment)
                result = result.rstrip('<|im_end|>')
                _candidates.add(result)
            _n_candidates = calc_and_filter(list(_candidates), _old_method, _new_method, _old_comment, params, _exp_comment)

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
            # output_s = json.dumps(output_dict, indent=2)
            # print(output_s)
            # writer.write(output_dict)

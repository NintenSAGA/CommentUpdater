import datetime
import pathlib

import jsonlines
import yaml
from dotenv import load_dotenv

from algo import calc_and_filter
from llm import Model

WORK_DIR = pathlib.Path(__file__).parent.parent.resolve()
CONFIG_DIR = WORK_DIR / 'config'

load_dotenv()


def result_preprocess(result: str):
    result = result.replace('<|im_end|>', '')
    result = result.replace("Updated:", '')
    result = result.strip().strip("'").strip('"').strip()
    result = result.lstrip('{').rstrip('}')

    result = result.strip()
    return result


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
            _old_comment = parsed['src_desc']
            _exp_comment = parsed['dst_desc']

            print(f'1️⃣Original: {_old_comment}')
            print(f'1️⃣Expected: {_exp_comment}')

            _n = params['nr_gen']
            _candidates = set()
            for i in range(_n):
                _result = myModel.resolve(_old_method, _new_method, _old_comment)
                _result = result_preprocess(_result)

                _candidates.add(_result)

            _n_candidates = calc_and_filter(
                candidates=list(_candidates), src_javadoc=_old_comment, params=params, exp_javadoc=_exp_comment)

            for cand in _n_candidates:
                print(
                    f'''🔸ED: {cand['ed']:.2f} RED: {cand['red']:.3f} GLEU: {cand['gleu']:.1f} METEOR: {cand['meteor']:.1f}
{cand['content']}''')

            output_dict = {
                'sample_id': parsed['sample_id'],
                'full_name': parsed['full_name'],
                'commit_id': parsed['commit_id'],
                'src_method': _old_method,
                'dst_method': _new_method,
                'src_desc': _old_comment,
                'dst_desc': _exp_comment,
            }

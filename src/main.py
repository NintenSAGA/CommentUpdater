import datetime
import pathlib

import jsonlines
import yaml
from dotenv import load_dotenv
from rich.progress import Progress

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


def generate_candidates(result_path: pathlib.Path) -> pathlib.Path:
    with open((CONFIG_DIR / 'config.yml'), 'r') as file:
        config = yaml.safe_load(file)
    test_data = config['testData']
    params = config['params']

    # myModel = Model('mistral-openorca')
    model_name = config['model']
    if 'rag_src' in config:
        rag_src = WORK_DIR / config['rag_src']
    else:
        rag_src = None
    my_model = Model(model_name, rag_src)
    print(f'Model: {model_name}')

    num = test_data['num']
    selected = test_data['selected']
    path = WORK_DIR / test_data['path']

    if num < 0:
        with open(path, mode='r') as f:
            num = len(f.readlines())

    with Progress() as progress:
        task = progress.add_task("[green]Generating...", total=num)
        progress.console.print(f'[green]{num} lines in total.')
        nr_line = 0
        with jsonlines.open(path, mode='r') as reader:
            with jsonlines.open(result_path, mode='w') as writer:
                for parsed in reader:
                    try:
                        _sample_id = parsed['sample_id']
                        if selected is not None and len(selected) != 0:
                            if _sample_id not in selected:
                                continue
                        elif nr_line == num:
                            break

                        _old_method = parsed['src_method']
                        _new_method = parsed['dst_method']
                        _old_comment = parsed['src_desc']
                        _exp_comment = parsed['dst_desc']

                        _n = params['nr_gen']
                        _candidates = []
                        for i in range(_n):
                            _result = my_model.resolve(_old_method, _new_method, _old_comment)
                            _result = result_preprocess(_result)

                            _candidates.append(_result)

                        output_dict = {
                            'SampleId': parsed['sample_id'],
                            'Origin': _old_comment,
                            'Reference': _exp_comment,
                            'LLMCandidates': _candidates
                        }

                        writer.write(output_dict)
                    except Exception as e:
                        progress.console.print(f'Error on line {nr_line}. Error: {e}')

                    progress.update(task, advance=1)
                    nr_line += 1
    return result_path


if __name__ == '__main__':
    result_file_name = f'candidates-{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl'
    result_fp = WORK_DIR / 'result' / 'candidates' / result_file_name
    result_fp = generate_candidates(result_fp)
    print(result_fp)

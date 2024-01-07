import json

import dotenv

from llm import LLM

if __name__ == '__main__':
    dotenv.load_dotenv()

    with open('./data/test.json', 'r') as raw_json:
        read = raw_json.read()
        parsed = json.loads(read)

    old_method = parsed['src_method']
    new_method = parsed['dst_method']
    old_comment = parsed['src_javadoc']

    print(f'''
Old Method: ```{old_method}```

New Method: ```{new_method}```

Old Comment: ```{old_comment}```

Please infer the new comment according to the code modification.
Rules: No explanation. No natural words. Perform the minimum modification Avoid any lose of information
''')

    # llm = LLM('codellama')
    llm = LLM('llama2')
    print(llm.infer_new_comment(old_comment, new_method, old_comment))

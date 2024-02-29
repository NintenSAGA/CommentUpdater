import javalang
import jsonlines
from javalang.parser import JavaSyntaxError

from algo import get_identifier_set

path = '../data/raw/test.jsonl'

line_cnt = 0
with jsonlines.open(path) as reader:
    for parsed in reader:
        line_cnt += 1
        old_method = parsed['src_method']
        new_method = parsed['dst_method']
        src_desc = parsed['src_desc']
        dst_desc = parsed['dst_desc']

        s = get_identifier_set(new_method)
        print(' '.join(s))

        exit(1)
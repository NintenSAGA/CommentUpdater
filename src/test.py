from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

path = '../data/raw/test.jsonl'

chat_model = ChatOpenAI(base_url="https://api.upstage.ai/v1/solar", model_name="solar-1-mini-chat",
                        api_key="ciBIuYCYRhZ1wx5xIod0dsnBj0hcpB2Q")
messages = [
    SystemMessage(
        content="You are a helpful assistant."
    ),
    HumanMessage(
        content="Hi, how are you?"
    )
]

response = chat_model.invoke(messages)
print(response)

exit(1)

line_cnt = 0
with jsonlines.open(path) as reader:
    for parsed in reader:
        line_cnt += 1
        if parsed['sample_id'] != 2336970:
            continue
        old_method = parsed['src_method']
        new_method = parsed['dst_method']
        src_desc = parsed['src_desc']
        dst_desc = parsed['dst_desc']

        print(old_method)
        print(new_method)

        s1 = get_identifier_set(new_method)
        s2 = get_identifier_set(old_method)
        s3 = s1 - s2
        print(s2)
        print(s1)
        print(s3)
        #
        # if line_cnt == 100:
        #     break

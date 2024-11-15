# ./test.json 의 한줄을 읽고 구조를 출력하느 코드

import json

with open('./test.json', 'r') as f:
    data = json.load(f)
    print(json.dumps(data[0], indent=4, ensure_ascii=False))

    with open('./ua-test.jsonl', 'w') as f:
        for row in data:
            f.write(json.dumps({
                'messages': [
                    {'role': 'user', 'content': row['content']},
                    {'role': 'assistant', 'content': row['headline']}
                ]
            }, ensure_ascii=False))

            if row != data[-1]:
                f.write('\n')

with open('./train.json', 'r') as f:
    data = json.load(f)
    print(json.dumps(data[0], indent=4, ensure_ascii=False))
    
    # 새로운 파일, AU-test.json를 생성하여 content를  messages: [{role: 'user', content: <content>}] 형식으로 저장
    # headline은 같은 행의 messages: assistant role의 content로 저장

    with open('./ua-train.jsonl', 'w') as f:
        for row in data:
            f.write(json.dumps({
                'messages': [
                    {'role': 'user', 'content': row['content']},
                    {'role': 'assistant', 'content': row['headline']}
                ]
            }, ensure_ascii=False))

            if row != data[-1]:
                f.write('\n')
import json

with open('A.json','r', encoding='utf-8') as f:
    A_data = json.load(f)

B_data = []
for item in A_data:
    new_item = {}
    new_item['instruction'] = '你现在是一个精通中国法律的法官,请对以下问题做出回答。' 
    new_item['input'] = item['input']
    new_item['answer'] = item['output']
    B_data.append(new_item)

with open('B.json','w',encoding='utf-8') as f:
  for item in B_data:
    json_str = json.dumps(item, ensure_ascii=False)
    f.write(json_str+'\n')
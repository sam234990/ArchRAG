import json
import pandas as pd
import re

# 打开并读取txt文件
with open('/mnt/data/wangshu/hcarag/MultiHop-RAG/questions/MultiHop-RAG_summary_questions_final.txt', 'r', encoding='utf-8') as file:
    content = file.read()

# 使用正则表达式提取问题
# questions_1 = re.findall(r'- \*\*Question \d+:\*\* (.+?)\s{2,}', content)
# questions_2 = re.findall(r'- Question \d+: (.+?)(?=\s{2,}|$)', content)
# questions = re.findall(r'- Question \d+: (.+?)(?=\n- Question \d+:|\n####|\n---|$)', content, re.DOTALL)
# questions = re.findall(r'- Question \d+: (.+?)(?=\n\s*- Question \d+:|\n\s*- Task|\n\s*- User|\n|$)', content, re.DOTALL)
# questions = re.findall(r'- Question \d+: (.+?)(?=\n\s*- Question \d+:|\n\s*- Task|\n\s*- User|\n|$)', content, re.DOTALL)
questions = re.findall(r'- Question \d+: (.+?)(?=\n\s*- Question \d+:|\n\s*- Task|\n\s*- User|\n|$)', content, re.DOTALL)

# questions = questions_1 + questions_2
# print(len(questions_1)+len(questions_2))
print(len(questions))
# quit()


# 将问题保存到JSON文件
questions_dict = [{"question": question, "answer":""} for question in questions]
tmp_df = pd.DataFrame(questions_dict)

save_path = '/mnt/data/wangshu/hcarag/MultiHop-RAG/questions/summary_Question.json'
tmp_df.to_json(save_path, orient='records', lines=True)


# print("Questions have been extracted and saved to questions.json")
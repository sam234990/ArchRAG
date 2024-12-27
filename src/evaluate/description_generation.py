# This code is based on the Query generation in LightRAG codebase.

from openai import OpenAI
import json
import time
import pandas as pd
import random

# os.environ["OPENAI_API_KEY"] = ""


def llm_invoker(
    input_text, temperature=0.7, max_tokens=8192, max_retries=5, json=False
):

    # api_key = "ollama"
    # # base_url = "http://localhost:11434/v1"
    # base_url = "http://localhost:5000/forward"
    # # base_url = "http://10.26.1.21:8877/v1"
    # # base_url = "http://10.26.1.186:6667/v1"
    # engine = "llama3.1:8b4k"
    
    api_key = "sk-AXOFue6Q3Tn9wEEP88Dc25C20d6549Da8d186557C9EcD7F9"
    base_url = "https://api.ai-gaochao.cn/v1"
    engine = "gpt-4o-2024-11-20"
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant that generates the description.",
        }
    ]
    message_prompt = {"role": "user", "content": input_text}
    messages.append(message_prompt)
    # 准备用于传递给 API 的参数字典
    parameters = {
        "model": engine,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    # 如果 json 为 True，加入 response_format 参数
    if json:
        parameters["response_format"] = {"type": "json_object"}

    print("start openai")
    retries = 0
    success = False
    result = None
    total_token = 0

    while not success and retries < max_retries:
        try:
            response = client.chat.completions.create(**parameters)
            result = response.choices[0].message.content
            total_token += response.usage.total_tokens
            success = True
        except Exception as e:
            retries += 1
            print(f"OpenAI error, retrying... ({retries}/{max_retries})")
            time.sleep(2)

    if not success:
        print("Failed after maximum retries.")
        # 可以抛出异常或返回 None
        raise Exception("Failed to get a response from OpenAI after multiple retries.")

    print("end openai")
    return result



if __name__ == "__main__":
    # MultiHop-RAG
    # dataset = "MultiHop-RAG"
    topic = "mix"
    # corpus = pd.read_json("/mnt/data/wangshu/hcarag/MultiHop-RAG/dataset/rag_multihop_corpus.json", lines=True, orient="records")
    corpus = pd.read_json(f"/mnt/data/wangshu/hcarag/ultradomain/{topic}.jsonl", lines=True, orient="records")
    random_list = random.sample(range(len(corpus)), 5)
    # print(random_list)
    # quit()
    max_length = 2048
    for i in random_list:
        if len(corpus.iloc[i]["context"]) <= 2048:
            max_length = len(corpus.iloc[i]["context"])

# mutyi-hop rag
#     prompt_generate_desciption = f"""Please generate the description of the news dataset according to the following contexts. The description should include the topic of this news dataset and the desciption shoud be more than 300 words. Note that the description should be informative, but not include the details of sampled contexts.
    
#     Here are the sampled contexts in this news dataset:
# context 1:
# {corpus.iloc[random_list[0]]["content"][:max_length]}
    
# context 2:
# {corpus.iloc[random_list[1]]["content"][:max_length]}
    
# context 3:
# {corpus.iloc[random_list[2]]["content"][:max_length]}
    
# context 4:
# {corpus.iloc[random_list[3]]["content"][:max_length]}
    
# context 5:
# {corpus.iloc[random_list[4]]["content"][:max_length]}
    
#     """

# 
    prompt_generate_desciption = f"""Please generate the description of the dataset according to the following contexts. The description should include the topic of this dataset and the desciption shoud be more than 300 words. Note that the description should be informative, but not include the details of sampled contexts.
    
    Here are the sampled contexts in this dataset:
context 1:
{corpus.iloc[random_list[0]]["context"][:max_length]}
    
context 2:
{corpus.iloc[random_list[1]]["context"][:max_length]}
    
context 3:
{corpus.iloc[random_list[2]]["context"][:max_length]}
    
context 4:
{corpus.iloc[random_list[3]]["context"][:max_length]}
    
context 5:
{corpus.iloc[random_list[4]]["context"][:max_length]}
    
    """
    print(prompt_generate_desciption)
    # quit()
    
    # description = "a QA dataset to evaluate retrieval and reasoning across documents with metadata in the RAG pipelines. It utilizes an English news article dataset as the underlying RAG knowledge base. It makes RAG benchmarking more closely resemble real-world scenarios and addresses the gap that existing RAG benchmarks not assess the retrieval and reasoning capability of LLMs for complex multi-hop queries. The MultiHop-RAG dataset contains six different types of news articles, covering 609 distinct news."
    
    description = llm_invoker(input_text=prompt_generate_desciption, json=False)
    # description = "aaaa"
    
    # file_path = f"/mnt/data/wangshu/hcarag/MultiHop-RAG/prompt/multihop_descriptipn_{time.time()}.txt"
    file_path = f"/mnt/data/wangshu/hcarag/ultradomain/prompt/{topic}/{topic}_descriptipn_{time.time()}.txt"
    with open(file_path, "w") as file:
        file.write(description)
    

    # result = llm_invoker(input_text=prompt_generate_desciption, json=False)


    print(f"Queries written to {file_path}")
    with open(file_path, "r") as file:
        read_description = file.read()
    print("the saved descrption:", read_description)
    
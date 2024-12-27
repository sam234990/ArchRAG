# This code is based on the Query generation in LightRAG codebase.

from openai import OpenAI
import json
import time
import pandas as pd


# os.environ["OPENAI_API_KEY"] = ""


def llm_invoker(
    input_text, temperature=0.7, max_tokens=16384, max_retries=5, json=False
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
            "content": "You are an AI assistant that helps people find information.",
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
    dataset = "agriculture"
    description_path = "/mnt/data/wangshu/hcarag/ultradomain/prompt/agriculture/agriculture_descriptipn_1735183681.2327023.txt"
    question_save_path = f"/mnt/data/wangshu/hcarag/ultradomain/question/agriculture/{dataset}_summary_questions_{time.time()}.txt"
    
    # description = "a QA dataset to evaluate retrieval and reasoning across documents with metadata in the RAG pipelines. It utilizes an English news article dataset as the underlying RAG knowledge base. It makes RAG benchmarking more closely resemble real-world scenarios and addresses the gap that existing RAG benchmarks not assess the retrieval and reasoning capability of LLMs for complex multi-hop queries. The MultiHop-RAG dataset contains six different types of news articles, covering 609 distinct news."
    
    
    
    description = ""
    
    with open(description_path, "r") as file:
        description = file.read()
    
    
    # # ultraDomain
    # # 1. argriculture
    # dataset = "ultraDomain_agriculture"
    # description = ""
    
    # # 2. cs
    # dataset = "ultraDomain_cs"
    # description = ""
    
    # # 3. legal
    # dataset = "ultraDomain_legal"
    # description = ""
    
    # # 4. mix
    # dataset = "ultraDomain_mix"
    # description = ""
    
    prompt = f"""
    Given the following description of a dataset:

    {description}

    Please identify 5 potential users who would engage with this dataset. For each user, list 5 tasks they would perform with this dataset. Then, for each (user, task) combination, generate 5 questions that require a high-level understanding of the entire dataset.

    Output the results in the following structure:
    - User 1: [user description]
        - Task 1: [task description]
            - Question 1:
            - Question 2:
            - Question 3:
            - Question 4:
            - Question 5:
        - Task 2: [task description]
            ...
        - Task 5: [task description]
    - User 2: [user description]
        ...
    - User 5: [user description]
        ...
    
    Note that there are 5 users and 5 tasks for each user, resulting in 25 tasks in total. Each task should have 5 questions, resulting in 125 questions in total.
    The Output should present the whole tasks and questions for each user.
    Output:
    """

    result = llm_invoker(input_text=prompt, json=False)

    file_path = f"./{dataset}_summary_questions_{time.time()}.txt"
    with open(file_path, "w") as file:
        file.write(result)

    print(f"Queries written to {file_path}")
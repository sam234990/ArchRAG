import json
import logging
import openai
from openai import OpenAI
import argparse
import time
from utils import create_arg_parser


log = logging.getLogger(__name__)

def llm_invoker(input_text, args, temperature=0.7, max_tokens=1500, max_retries=5):

    if "llama" in args.engine.lower():
        api_key = "ollama"
        base_url = args.api_base
    else:
        api_key = args.api_key
        base_url = args.api_base

    engine = args.engine
    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant that helps people find information.",
        }
    ]
    message_prompt = {"role": "user", "content": input_text}
    messages.append(message_prompt)
    print("start openai")
    retries = 0
    success = False
    result = None

    while not success and retries < max_retries:
        try:
            response = client.chat.completions.create(
                model=engine,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=0,
                presence_penalty=0,
            )
            result = response.choices[0].message.content
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
    parser = create_arg_parser()
    
    # 解析参数
    args = parser.parse_args()

    test_input_text = "What is the capital of France?"
    result = llm_invoker(test_input_text, args)
    print(f"llm_invoker result: {result}")

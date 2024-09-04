import json

from openai import OpenAI


def ollama_response():
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # required, but unused
    )

    response = client.chat.completions.create(
        model="llama3.1:8b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {"role": "assistant", "content": "The LA Dodgers won in 2020."},
            {"role": "user", "content": "Where was it played?"},
        ],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    ollama_response()

from src.model.llm import LLM

load_model = {
    "llm": LLM,
    "inference_llm": LLM,
}

# Replace the following with the model paths
llama_model_path = {
    "7b": "meta-llama/Llama-2-7b-hf",
    "8b": "/mnt/data/grouph_share/transformer_llm/Meta-Llama-3.1-8B-Instruct",
    "7b_chat": "meta-llama/Llama-2-7b-chat-hf",
    "13b": "meta-llama/Llama-2-13b-hf",
    "13b_chat": "meta-llama/Llama-2-13b-chat-hf",
}

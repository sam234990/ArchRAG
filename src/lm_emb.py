from tqdm import tqdm
import gensim
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
import numpy as np
import openai
from openai import OpenAI


pretrained_repo = "/mnt/data/wangshu/all-roberta-large-v1"
batch_size = 256  # Adjust the batch size as needed


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_ids=None, attention_mask=None):
        super().__init__()
        self.data = {
            "input_ids": input_ids,
            "att_mask": attention_mask,
        }

    def __len__(self):
        return self.data["input_ids"].size(0)

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()
        batch_data = dict()
        for key in self.data.keys():
            if self.data[key] is not None:
                batch_data[key] = self.data[key][index]
        return batch_data


class Sentence_Transformer(nn.Module):

    def __init__(self, pretrained_repo):
        super(Sentence_Transformer, self).__init__()
        print(f"inherit model weights from {pretrained_repo}")
        self.bert_model = AutoModel.from_pretrained(pretrained_repo)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        data_type = token_embeddings.dtype
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(data_type)
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(self, input_ids, att_mask):
        bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
        sentence_embeddings = self.mean_pooling(bert_out, att_mask)

        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings


def load_sbert():

    model = Sentence_Transformer(pretrained_repo)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_repo)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    return model, tokenizer, device


def sber_text2embedding(model, tokenizer, device, text):
    try:
        encoding = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        dataset = Dataset(
            input_ids=encoding.input_ids, attention_mask=encoding.attention_mask
        )

        # DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Placeholder for storing the embeddings
        all_embeddings = []

        # Iterate through batches
        with torch.no_grad():

            for batch in dataloader:
                # Move batch to the appropriate device
                batch = {key: value.to(device) for key, value in batch.items()}

                # Forward pass
                embeddings = model(
                    input_ids=batch["input_ids"], att_mask=batch["att_mask"]
                )

                # Append the embeddings to the list
                all_embeddings.append(embeddings)

        # Concatenate the embeddings from all batches
        all_embeddings = torch.cat(all_embeddings, dim=0).cpu()

    except:
        return torch.zeros((0, 1024))

    return all_embeddings


def load_contriever():
    print("Loading contriever model...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
    model = AutoModel.from_pretrained("facebook/contriever")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)
    model.eval()
    return model, tokenizer, device


def contriever_text2embedding(model, tokenizer, device, text):

    def mean_pooling(token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    try:
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        dataset = Dataset(
            input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
        )

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_embeddings = []
        with torch.no_grad():
            for batch in dataloader:
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"], attention_mask=batch["att_mask"]
                )
                embeddings = mean_pooling(outputs[0], batch["att_mask"])
                all_embeddings.append(embeddings)
            all_embeddings = torch.cat(all_embeddings, dim=0).cpu()
    except:
        all_embeddings = torch.zeros((0, 1024))

    return all_embeddings


def openai_embedding(input_text, api_key, api_base, engine="text-embedding-ada-002"):
    """
    Generate an embedding for the input text using OpenAI's API.

    Args:
        input_text (str): The text for which to generate the embedding.
        api_key (str): Your OpenAI API key.
        api_base (str): The base URL of the OpenAI API.
        engine (str): The embedding model engine to use (default: 'text-embedding-ada-002').

    Returns:
        list: The embedding vector for the input text.
    """
    try:
        # Set up OpenAI API credentials and base URL
        # TODO: The 'openai.api_base' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(base_url=api_base)'
        # openai.api_base = api_base
        client = OpenAI(api_key=api_key, base_url=api_base)

        # Request the embedding from the OpenAI API
        response = client.embeddings.create(input=input_text, model=engine)

        # Extract the embedding from the response
        embedding = response.data[0].embedding
        return embedding

    except Exception as e:
        print(f"Failed to generate embedding: {e}")
        return None


load_model = {
    "sbert": load_sbert,
    "contriever": load_contriever,
}


load_text2embedding = {
    "sbert": sber_text2embedding,
    "contriever": contriever_text2embedding,
}


if __name__ == "__main__":
    test_text = "What is the capital of France?"
    test_embedding = openai_embedding(
        input_text=test_text,
        api_key="ollama",
        api_base="http://localhost:11434/v1",
        engine="nomic-embed-text",
    )
    print("the shape of the embedding is: ", len(test_embedding))
    print("the first 10 elements of the embedding are: ", test_embedding[:10])

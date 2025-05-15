# ArchRAG

This repository contains code and data processing for the paper "ArchRAG: Attributed Community-based Hierarchical
Retrieval-Augmented Generation"

ArchRAG is a novel graph-based RAG approach by using attributed communities organized hierarchically, and introduce a novel LLM-based hierarchical clustering method.
For more details, check out our paper.


## Setup Environment

This project implements C-HNSW using a custom Faiss framework. Follow the steps below to set up the environment correctly.


### 1. Create a Python 3.10 Environment
We recommend using conda to manage the environment:

```bash
conda create -n archrag python=3.10 -y
conda activate archrag
```

### 2. Install Dependencies
Install the required Python packages using:
```bash
pip install -r requirements.txt
```
### 3. Install Custom Faiss for C-HNSW
The C-HNSW component requires a modified version of Faiss. Please refer to this [README](./HCHNSW/README.md) for installation instructions.

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```

## Running ArchRAG

Using our ArchRAG framework requires a two-step, offline index and online retrieval.

### Offline Index

Before constructing ArchRAG index, we first use Microsoft GraphRAG to extract KG from corpus, please refer to the source code and [instruction](./src/graphrag/README.md).

We provide a bash for constructing ArchRAG index.

```shell
bash dataset/index.sh
```

### Online Retrieval

We provide a bash for online retrieval given a specific dataset.

```shell
bash dataset/query.sh
```

## Data format and Environment


Corpus

```json
{
"title": "FIRST TITLE",
"context": "FIRST TEXT",
"id": 0
}
{
"title": "SECOND TITLE",
"context": "SECOND TEXT",
"id": 1
}
```

Question

```json
{
"question": "QUESTION 1",
"options": "DICT-style options for multiple-choice questions (Optional)",
"answer": "ANSWER",
"answer_idx":"Answer options for multiple-choice questions (Optional)",
"id": 0
}
```

One can use GraphRAG to construct the Knowledge graph and use the "final_entity" and "final_relationship" file.

## Microsoft GraphRAG baseline and graph construction

This repository is a baseline comparison (GraphRAG --GLobal Search \& Local Search) and graph construction implementation for ArchRAG, based on the Microsoft GraphRAG repository (v0.3.0).

For more information, please refer to the original repository: [Microsoft GraphRAG](https://github.com/microsoft/GraphRAG).

### Run Instructions

1. Install dependencies:
   ```bash
    pip install graphrag
   ```

2. Create a folder for corpus
   ```bash
    mkdir -p ./corpus/input
   ```


3. Run the initialization:
   ```bash
   python -m graphrag.index --init --root ./corpus
   ```

4. Modify the setting.yaml and Create index by GraphRAG
   ```bash
    python -m graphrag.index --root ./corpus
   ```

   For graph construction in ArchRAG, you should first modify the setting.yaml in the corpus by adding skip workflow (including "create_final_community_reports", "create_summarized_entities", and e.t.c.). An example setting.yaml are provided in [setting.yaml](../../dataset/settings.yaml) 

5. Copy the constructed graph(Entity \& Relationship file) for ArchRAG
   ```bash
    mkdir -p ./archrag/index/ 
    cp ./corpus/output/TIMESTAMP/artifacts/create_final_relationships.parquet ./archrag/index/
    cp ./corpus/output/TIMESTAMP/artifacts/create_final_entities.parquet ./archrag/index/  
   ```
   Note that "TIMESTAMP" is the time for creating graph

For detailed usage and configuration, refer to the original GraphRAG documentation: [Microsoft GraphRAG](https://github.com/microsoft/GraphRAG).
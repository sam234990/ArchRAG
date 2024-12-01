data=$1  # e.g., 'sample'
retriever_name=$2  # e.g., 'facebook/contriever'
extraction_model=$3 # e.g., 'gpt-3.5-turbo-1106' (OpenAI), 'meta-llama/Llama-3-8b-chat-hf' (Together AI)
available_gpus=$4
syn_thresh=$5 # float, e.g., 0.8
llm_api=$6 # e.g., 'openai', 'together'
extraction_type=ner


# 记录总开始时间
start_time=$(date +%s)

# Running Open Information Extraction
python src/openie_with_retrieval_option_parallel.py --dataset $data --llm $llm_api \
    --model_name $extraction_model --run_ner --num_passages all # NER and OpenIE for passages
python src/named_entity_extraction_parallel.py --dataset $data --llm $llm_api \
    --model_name $extraction_model  # NER for queries

# Creating Contriever Graph
python src/create_graph.py --dataset $data --model_name $retriever_name \
    --extraction_model $extraction_model --threshold $syn_thresh --extraction_type $extraction_type \
    --cosine_sim_edges

# Getting Nearest Neighbor Files
CUDA_VISIBLE_DEVICES=$available_gpus python src/RetrievalModule.py --retriever_name $retriever_name \
    --string_filename output/query_to_kb.tsv
CUDA_VISIBLE_DEVICES=$available_gpus python src/RetrievalModule.py --retriever_name $retriever_name \
    --string_filename output/kb_to_kb.tsv
CUDA_VISIBLE_DEVICES=$available_gpus python src/RetrievalModule.py --retriever_name $retriever_name \
    --string_filename output/rel_kb_to_kb.tsv

python src/create_graph.py --dataset $data --model_name $retriever_name \
    --extraction_model $extraction_model --threshold $syn_thresh --create_graph \
    --extraction_type $extraction_type --cosine_sim_edges



# 记录总结束时间
end_time=$(date +%s)
total_time=$((end_time - start_time))

# 输出总运行时间
echo "Total runtime: $total_time seconds"

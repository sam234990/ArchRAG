#!/bin/bash

eval_mode="DocQA"

api_key="" #TODO
api_base="" #TODO
engine="llama3.1:8b4k" # llm engine

output_dir="archrag_index" # output index path
base_path="archrag" # input dataset path 

relationship_filename="create_final_relationships.parquet"
entity_filename="create_final_entities.parquet"

dataset_name="multihop"

strategy="global"
k_each_level=5
topk_chunk=0
k_final=15
topk_e=10
all_k_inference=15
generate_strategy="mr"
response_type="QA"

temperature=0.1
only_entity=False
ppr_refine=False
involve_llm_res=True

num_workers=10

python_file="src/evaluate/test_qa.py"



log_file="eval.log"

nohup python -u $python_file --strategy $strategy --k_each_level $k_each_level \
    --k_final $k_final --all_k_inference $all_k_inference --topk_e $topk_e \
    --generate_strategy $generate_strategy --response_type $response_type \
    --temperature $temperature --eval_mode $eval_mode \
    --output_dir $output_dir --base_path $base_path --dataset_name $dataset_name \
    --relationship_filename $relationship_filename --entity_filename $entity_filename \
    --only_entity $only_entity --ppr_refine $ppr_refine \
    --involve_llm_res $involve_llm_res --topk_chunk $topk_chunk  \
    --engine $engine --num_workers $num_workers \
    --api_key $api_key --api_base $api_base \
    >$log_file 2>&1 &

echo "log file: $log_file"

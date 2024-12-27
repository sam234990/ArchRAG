#!/bin/bash

source activate rag

eval_mode="DocQA"

output_dir="/mnt/data/wangshu/hcarag/MultiHop-RAG/hcarag/hc_index_8b"
base_path="/mnt/data/wangshu/hcarag/MultiHop-RAG/hcarag"

relationship_filename="create_final_relationships.parquet"
entity_filename="create_final_entities.parquet"

dataset_name="multihop"

strategy="global"
# k_each_level=5
k_final=15
topk_e=10
all_k_inference=15
# generate_strategy="direct"
generate_strategy="mr"
response_type="QA"

temperature=0.1
only_entity=True
ppr_refine=False
involve_llm_res=False
# topk_chunk=2

num_workers=20

python_file="/home/wangshu/rag/hier_graph_rag/src/evaluate/test_qa.py"

# 设置PYTHONPATH
export PYTHONPATH="/home/wangshu/rag/hier_graph_rag/:$PYTHONPATH"


# k_each_level=5
# topk_chunk=2



# for chunk_k in {2..3}; do
#     topk_chunk=$chunk_k
#     # for level_k in {5..10}; do
#     for level_k in {2..4}; do
#         k_each_level=$level_k
#         echo "topk_chunk: $topk_chunk, k_each_level: $k_each_level"

# log_file="./eval/usage_evaluate_t${temperature}_${strategy}_${k_each_level}_"\
# "${k_final}_${topk_e}_${all_k_inference}_${generate_strategy}_${topk_chunk}_"\
# "inv${involve_llm_res}_oe${only_entity}.log"
        
#         echo "log file: $log_file"
#         python -u $python_file --strategy $strategy --k_each_level $k_each_level \
#             --k_final $k_final --all_k_inference $all_k_inference --topk_e $topk_e \
#             --generate_strategy $generate_strategy --response_type $response_type \
#             --temperature $temperature --eval_mode $eval_mode \
#             --output_dir $output_dir --base_path $base_path --dataset_name $dataset_name \
#             --relationship_filename $relationship_filename --entity_filename $entity_filename \
#             --only_entity $only_entity --num_workers $num_workers --ppr_refine $ppr_refine \
#             --involve_llm_res $involve_llm_res --topk_chunk $topk_chunk  \
#             >$log_file 2>&1 &
#         wait
#     done
# done

k_each_level=5
topk_chunk=2



# augment_graph=False
augment_graph=True
# cluster_method="weighted_leiden"
cluster_method="spectral"

output_dir="/mnt/data/wangshu/hcarag/MultiHop-RAG/hcarag/index_${augment_graph}_${cluster_method}"


log_file="./variant/index_${augment_graph}_${cluster_method}_usage_evaluate_t${temperature}_${strategy}_${k_each_level}_"\
"${k_final}_${topk_e}_${all_k_inference}_${generate_strategy}_${topk_chunk}_"\
"inv${involve_llm_res}_oe${only_entity}.log"

nohup python -u $python_file --strategy $strategy --k_each_level $k_each_level \
    --k_final $k_final --all_k_inference $all_k_inference --topk_e $topk_e \
    --generate_strategy $generate_strategy --response_type $response_type \
    --temperature $temperature --eval_mode $eval_mode \
    --output_dir $output_dir --base_path $base_path --dataset_name $dataset_name \
    --relationship_filename $relationship_filename --entity_filename $entity_filename \
    --only_entity $only_entity --num_workers $num_workers --ppr_refine $ppr_refine \
    --involve_llm_res $involve_llm_res --topk_chunk $topk_chunk  \
    >$log_file 2>&1 &

echo "log file: $log_file"

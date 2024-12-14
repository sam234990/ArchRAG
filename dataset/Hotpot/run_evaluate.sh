conda activate rag

strategy="global"
k_each_level=7
k_final=15
topk_e=10
all_k_inference=15
# generate_strategy="direct"
generate_strategy="mr"
response_type="QA"

temperature=0.1
only_entity=True
ppr_refine=False

involve_llm_res=True

num_workers=20

eval_mode="DocQA"

# output_dir="/mnt/data/wangshu/hcarag/HotpotQA/hcarag/hc_index_8b"
output_dir="/mnt/data/wangshu/hcarag/HotpotQA/hcarag/hc_index_8b4k"
base_path="/mnt/data/wangshu/hcarag/HotpotQA/hcarag"

relationship_filename="create_final_relationships.parquet"
entity_filename="create_final_entities.parquet"

dataset_name="hotpot"

log_file="./eval/usage_evaluate_t${temperature}_${strategy}_${k_each_level}_""\
${k_final}_${topk_e}_${all_k_inference}_${generate_strategy}_""\
${response_type}_inv${involve_llm_res}_oe${only_entity}.log"

python_file="/home/wangshu/rag/hier_graph_rag/src/evaluate/test_qa.py"

# 设置PYTHONPATH
export PYTHONPATH="/home/wangshu/rag/hier_graph_rag/:$PYTHONPATH"

nohup python -u $python_file --strategy $strategy --k_each_level $k_each_level \
    --k_final $k_final --all_k_inference $all_k_inference --topk_e $topk_e \
    --generate_strategy $generate_strategy --response_type $response_type \
    --temperature $temperature --eval_mode $eval_mode \
    --output_dir $output_dir --base_path $base_path --dataset_name $dataset_name \
    --relationship_filename $relationship_filename --entity_filename $entity_filename \
    --only_entity $only_entity --num_workers $num_workers --ppr_refine $ppr_refine \
    --involve_llm_res $involve_llm_res \
    >$log_file 2>&1 &

echo "log file: $log_file"

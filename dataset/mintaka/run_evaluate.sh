strategy="global"
k_each_level=10
k_final=20
topk_e=10
all_k_inference=15
# generate_strategy="mr"
generate_strategy="direct"
response_type="QA"

temperature=0.1
only_entity=False
num_workers=24

output_dir="/mnt/data/wangshu/hcarag/mintaka/hcarag/hc_index_8b"
base_path="/mnt/data/wangshu/hcarag/mintaka/KG"
dataset_path="/mnt/data/wangshu/hcarag/mintaka/QA/mintaka_test_qa.json"

log_file="./eval/evaluate_t${temperature}_${strategy}_${k_each_level}_${k_final}_${topk_e}_${all_k_inference}_${generate_strategy}_${response_type}.log"
# log_file="./eval/test_t${temperature}_${strategy}_${k_each_level}_${k_final}_${topk_e}_${all_k_inference}_${generate_strategy}_${response_type}.log"
python_file="/home/wangshu/rag/hier_graph_rag/src/evaluate/test_qa.py"

# 设置PYTHONPATH
export PYTHONPATH="/home/wangshu/rag/hier_graph_rag/:$PYTHONPATH"

nohup python -u $python_file --strategy $strategy --k_each_level $k_each_level \
    --k_final $k_final --all_k_inference $all_k_inference --topk_e $topk_e \
    --generate_strategy $generate_strategy --response_type $response_type \
    --output_dir $output_dir --base_path $base_path --dataset_path $dataset_path \
    --temperature $temperature \
    --only_entity $only_entity --num_workers $num_workers \
    >$log_file 2>&1 &

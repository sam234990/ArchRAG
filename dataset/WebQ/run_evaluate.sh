strategy="global"
k_each_level=10
k_final=15
topk_e=20
all_k_inference=15
# generate_strategy="direct"
generate_strategy="mr"
response_type="QA"

temperature=0.7
only_entity=False
ppr_refine=False

num_workers=24


output_dir="/mnt/data/wangshu/hcarag/FB15k/hc_index_8b"
base_path="/mnt/data/wangshu/hcarag/FB15k/KG"

dataset_name="webq"


log_file="./eval/evaluate_t${temperature}_${strategy}_${k_each_level}_\
${k_final}_${topk_e}_${generate_strategy}_${response_type}_${only_entity}_${ppr_refine}.log"
python_file="/home/wangshu/rag/hier_graph_rag/src/evaluate/test_qa.py"

# 设置PYTHONPATH
export PYTHONPATH="/home/wangshu/rag/hier_graph_rag/:$PYTHONPATH"

nohup python -u $python_file --strategy $strategy --k_each_level $k_each_level \
    --k_final $k_final --all_k_inference $all_k_inference --topk_e $topk_e \
    --generate_strategy $generate_strategy --response_type $response_type \
    --output_dir $output_dir --base_path $base_path --dataset_name $dataset_name \
    --temperature $temperature --num_workers $num_workers \
    --only_entity $only_entity --ppr_refine $ppr_refine \
    >$log_file 2>&1 &

echo "log file: $log_file"
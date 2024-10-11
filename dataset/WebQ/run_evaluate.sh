strategy="global"
k_each_level=10
k_final=15
topk_e=10
all_k_inference=15
generate_strategy="direct"
# generate_strategy="mr"
response_type="QA"
temperature=0.1
only_entity=True
num_workers=24


log_file="./eval/test1_t${temperature}_${strategy}_${k_each_level}_${k_final}_${topk_e}_${all_k_inference}_${generate_strategy}_${response_type}.log"
python_file="/home/wangshu/rag/hier_graph_rag/src/evaluate/test_qa.py"

# 设置PYTHONPATH
export PYTHONPATH="/home/wangshu/rag/hier_graph_rag/:$PYTHONPATH"

nohup python -u $python_file --strategy $strategy --k_each_level $k_each_level \
    --k_final $k_final --all_k_inference $all_k_inference --topk_e $topk_e \
    --generate_strategy $generate_strategy --response_type $response_type \
    --temperature $temperature \
    --only_entity $only_entity --num_workers $num_workers \
    >$log_file 2>&1 &

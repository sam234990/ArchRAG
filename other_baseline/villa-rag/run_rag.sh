dataset_name="hotpot"
# dataset_name="multihop"
# dataset_name="narrativeqa"


engine="llama3.1:8b4k"
# engine="llama2:7b"
topk=1
log_file="./eval_res/usage-${dataset_name}-${topk}-${engine}.log"
python_file="./villa_rag.py"
eval_mode="DocQA"
num_workers=8

# 设置PYTHONPATH
export PYTHONPATH="/home/wangshu/rag/hier_graph_rag:$PYTHONPATH"

nohup python -u $python_file --dataset_name $dataset_name --topk $topk \
    --eval_mode $eval_mode --engine $engine --num_workers $num_workers \
    >$log_file 2>&1 &

echo "log file: $log_file"

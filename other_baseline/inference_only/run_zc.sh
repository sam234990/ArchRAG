# dataset_name="hotpot"
# dataset_name="multihop"
# dataset_name="webq"
dataset_name="narrativeqa"
# dataset_name="mintaka"
# dataset_name="webqsp"

# strategy="cot"
strategy="zero-shot"

engine="llama3.1:8b4k"
# engine="llama2:7b"

log_file="./eval_res/usage-${dataset_name}-${strategy}-${engine}.log"
python_file="./zero-cot.py"
# eval_mode="DocQA"
eval_mode="KGQA"
num_workers=20

# 设置PYTHONPATH
export PYTHONPATH="/home/wangshu/rag/hier_graph_rag:$PYTHONPATH"

nohup python -u $python_file --dataset_name $dataset_name --strategy $strategy \
    --eval_mode $eval_mode --engine $engine --num_workers $num_workers \
    >$log_file 2>&1 &

echo "log file: $log_file"

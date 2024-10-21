# dataset_name="hotpot"
# dataset_name="multihop"
dataset_name="webq"
# dataset_name="mintaka"
strategy="cot"
# strategy="zero-shot"
log_file="./eval_res/inference_only-${dataset_name}-${strategy}.log"
python_file="./zero-cot.py"
# eval_mode="DocQA"
eval_mode="KGQA"

# 设置PYTHONPATH
export PYTHONPATH="/home/wangshu/rag/hier_graph_rag:$PYTHONPATH"

nohup python -u $python_file --dataset_name $dataset_name --strategy $strategy \
    --eval_mode $eval_mode \
    >$log_file 2>&1 &

echo "log file: $log_file"
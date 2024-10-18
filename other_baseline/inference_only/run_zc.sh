dataset_name="hotpot"
strategy="cot"
# strategy="zero-shot"
log_file="./eval_res/inference_only-${dataset_name}-${strategy}.log"
python_file="./zero-cot.py"
eval_mode="DocQA"

# 设置PYTHONPATH
export PYTHONPATH="/home/wangshu/rag/hier_graph_rag:$PYTHONPATH"

nohup python -u $python_file --dataset_name $dataset_name --strategy $strategy \
    --eval_mode $eval_mode \
    >$log_file 2>&1 &

echo "log file: $log_file"
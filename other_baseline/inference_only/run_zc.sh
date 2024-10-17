dataset_name="mintaka"
strategy="cot"
log_file="./eval_res/inference_only-${strategy}_${dataset_name}.log"
python_file="./zero-cot.py"

# 设置PYTHONPATH
export PYTHONPATH="/home/wangshu/rag/hier_graph_rag:$PYTHONPATH"

nohup python -u $python_file --dataset_name $dataset_name --strategy $strategy \
    > $log_file 2>&1 &

log_file="./eval_res/inference_only-zero_mintaka.log"
python_file="./zero-cot.py"

# 设置PYTHONPATH
export PYTHONPATH="/home/wangshu/rag/hier_graph_rag:$PYTHONPATH"

nohup python -u $python_file >$log_file 2>&1 &

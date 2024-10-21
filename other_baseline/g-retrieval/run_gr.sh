temperature=0.1
# dataset="webq"
dataset="mintaka"

log_file="./inf_res/gr_${dataset}_${temperature}.log"
python_file="/home/wangshu/rag/hier_graph_rag/other_baseline/g-retrieval/inference_only_llm.py"

# 设置PYTHONPATH
export PYTHONPATH="/home/wangshu/rag/hier_graph_rag/other_baseline/g-retrieval/:$PYTHONPATH"

nohup python -u $python_file --temperature $temperature --dataset $dataset >$log_file 2>&1 &

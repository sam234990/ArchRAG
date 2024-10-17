temperature=0.1

log_file="./inference_res/gr_webq_${temperature}.log"
python_file="/home/wangshu/rag/hier_graph_rag/other_baseline/g-retrieval/inference_only_llm.py"

# 设置PYTHONPATH
export PYTHONPATH="/home/wangshu/rag/hier_graph_rag/other_baseline/g-retrieval/:$PYTHONPATH"

nohup python -u $python_file --temperature $temperature >$log_file 2>&1 &

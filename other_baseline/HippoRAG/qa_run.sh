# 启动 HippoRAG 环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate hipporag

DATA=hotpotqa
python_file="./src/qa/qa_reader.py"
log_file="logs/${DATA}_retrieval.log"

nohup python $python_file --dataset hotpotqa --retriever colbertv2 \
    --data output/ircot/ircot_results_hotpotqa_hipporag_colbertv2_demo_0_llama3_1:8b4k_no_ensemble_step_1_top_10_sim_thresh_0.8_damping_0.5.json \
    >$log_file 2>&1 &

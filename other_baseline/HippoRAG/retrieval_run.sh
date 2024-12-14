# 启动 HippoRAG 环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate hipporag

# DATA=hotpotqa
DATA=multihop
log_file="logs/${DATA}_retrieval.log"

nohup bash src/run_hipporag_main_exps.sh \
    > $log_file 2>&1 &

DATA=multihop
# DATA=hotpotqa
LLM=llama3.1:8b4k
SYNONYM_THRESH=0.8
GPUS=0
LLM_API=ollama # LLM API provider e.g., 'openai', 'together', see 'src/langchain_util.py'

log_file="logs/${DATA}_index.log"

nohup bash src/setup_hipporag_colbert.sh $DATA $LLM $GPUS $SYNONYM_THRESH $LLM_API \
    >$log_file 2>&1 &

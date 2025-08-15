export OPENAI_API_KEY=1935636181065207882
# 1648285843187896359
export OPENAI_API_BASE=https://aigc.sankuai.com/v1/openai/native/chat/completions
export TORCH_DISTRIBUTED_TIMEOUT=53600

eval "$('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/lanxiaohan/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate qwen2vl_lxh

kill -f -9 ray

export LMUData=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/qiuhaibo/workspace/projects/eval/vlm_eval/vlmeval/data_root
export https_proxy=http://10.229.18.23:3128
export http_proxy=http://10.229.18.23:3128

cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lanxiaohan/reasoning_hc/code/VLMEvalKit/

### mutlinode variables ###
cluster_spec=${AFO_ENV_CLUSTER_SPEC//\"/\\\"}
echo "cluster spec is $cluster_spec"
worker_list_command="import json_parser;print(json_parser.parse(\"$cluster_spec\", \"worker\"))"
echo "worker list command is $worker_list_command"
eval worker_list=`python -c "$worker_list_command"`
worker_strs=(${worker_list//,/ })
master=${worker_strs[0]}
echo "master is $master"
master_strs=(${master//:/ })
master_addr=${master_strs[0]}
master_port=${master_strs[1]}
echo "master address is $master_addr"
echo "master port is $master_port"
dist_url="tcp://$master_addr:$master_port"

index_command="import json_parser;print(json_parser.parse(\"$cluster_spec\", \"index\"))"
eval node_rank=`python -c "$index_command"`
echo "node rank is $node_rank"

master_node=${worker_strs[0]}
echo "master node is $master_node"

CKPT=$1
STEP=$2
MODEL_TYPE=${3:-"Qwen2.5-VL-7B-Instruct-Eureka-CKPT-ReTool"} # set it should also specify step!
OUTPUT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lanxiaohan/qwenvl25_outputs/"

# 数据集分配映射
declare -A dataset_mapping
dataset_mapping[0]="LogicVista"
dataset_mapping[1]="MathVision_CODE_YES MathVista_MINI"
dataset_mapping[2]="MathVerse_MINI_Vision_Only_CODE_YES MathVerse_MINI_Vision_Only"
dataset_mapping[3]="MathVision"

# 检查node_rank是否有效
if [[ ! ${dataset_mapping[$node_rank]+_} ]]; then
    echo "Error: NODE_RANK should be 0, 1, 2, or 3"
    exit 1
fi

# 获取当前node对应的数据集
part_dataset=(${dataset_mapping[$node_rank]})

echo "Node $node_rank processing datasets: ${part_dataset[@]}"

# 遍历分配给当前node的数据集
for dataset in "${part_dataset[@]}"; do
    echo "Processing dataset: $dataset on node $node_rank"
    echo "ckpt is $CKPT, step is ${STEP:-unset}"
    # 在这里添加你的处理逻辑
    if [ -n "$STEP" ]; then
        # 带checkpoint步数的模式
        work_dir="${OUTPUT_DIR}/vlmeval_results_0509/${CKPT}-checkpoint-${STEP}"
        ckpt_path="${OUTPUT_DIR}/${CKPT}/full/sft/checkpoint-${STEP}"
    else
        echo "no step(not supported now!)"
        exit 1
        # 不带checkpoint步数的模式
        # work_dir="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/vacv-data/mlm/qwenvl25_outputs/vlmeval_results/${CKPT}"
        # ckpt_path="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/vacv-data/mlm/qwenvl25_outputs/${CKPT}/full/sft"
    fi

    # 执行训练命令
    export TOKENIZERS_PARALLELISM=false
    torchrun --nproc-per-node=8 --nnodes=1 \
        run.py --data "$dataset" \
        --work-dir "$work_dir" \
        --model $MODEL_TYPE \
        --ckpt "$ckpt_path" \
        --reuse
done

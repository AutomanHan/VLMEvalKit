export OPENAI_API_KEY=1648285843187896359
export OPENAI_API_BASE=https://aigc.sankuai.com/v1/openai/native/chat/completions

eval "$('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/lanxiaohan/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate qwen2vl_lxh

kill -f -9 ray

export LMUData=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/qiuhaibo/workspace/projects/eval/vlm_eval/vlmeval/data_root
export https_proxy=http://10.229.18.23:3128
export http_proxy=http://10.229.18.23:3128

cd /mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/lanxiaohan/mig_from_vacv/projects/mm_large_model/for_o1/VLMEvalKit_0509/VLMEvalKit/

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
OUTPUT_DIR=${3:-"/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lanxiaohan/qwenvl25_outputs/"}

echo "ckpt is $CKPT, step is ${STEP:-unset}"
for dataset in MathVista_MINI MathVerse_MINI_Vision_Only MathVision DynaMath WeMath LogicVista; do
# for dataset in MathVision MathVista_MINI MathVerse_MINI_Vision_Only DynaMath WeMath LogicVista; do
    echo "Processing dataset: $dataset"
    # 根据是否传入STEP参数选择不同的路径模式
    if [ -n "$STEP" ]; then
        # 带checkpoint步数的模式
        work_dir="${OUTPUT_DIR}/vlmeval_results_0509/${CKPT}-checkpoint-${STEP}"
        ckpt_path="${OUTPUT_DIR}/${CKPT}/full/sft/checkpoint-${STEP}"
    else
        # echo "no step(not supported now!)"
        # exit 1
        # 不带checkpoint步数的模式
        work_dir="${OUTPUT_DIR}/vlmeval_results_0509/${CKPT}"
        ckpt_path="${OUTPUT_DIR}/${CKPT}/full/sft"
        # work_dir="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/vacv-data/mlm/qwenvl25_outputs/vlmeval_results/${CKPT}"
        # ckpt_path="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/vacv-data/mlm/qwenvl25_outputs/${CKPT}/full/sft"
    fi

    # 执行训练命令
    torchrun --nproc-per-node=8 --nnodes=1 \
        run.py --data "$dataset" \
        --work-dir "$work_dir" \
        --model Qwen2.5-VL-7B-Instruct-Eureka-CKPT \
        --ckpt "$ckpt_path" \
        --reuse
done

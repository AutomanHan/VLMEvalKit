export OPENAI_API_KEY=1648285843187896359
export OPENAI_API_BASE=https://aigc.sankuai.com/v1/openai/native/chat/completions

eval "$('/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/mllm_env/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate mm-eureka-hl
echo "conda activate mm-eureka-hl"

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

for dataset in MathVerse_MINI_Vision_Only; do
# for dataset in DynaMath LogicVista WeMath; do
# for dataset in MathVision MathVerse_MINI MathVista_MINI OlympiadBench_MINI R1_OneVision_V2; do
    echo "dataset is $dataset"
    torchrun --nproc-per-node=8 --nnodes=1 --node_rank=${node_rank} --master_addr=${master_addr} --master-port=25580 \
        run.py --data $dataset \
        --work-dir /mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/lanxiaohan/qwenvl25_outputs/vlmeval_results_0509_version/ \
        --model Qwen2.5-VL-7B-Instruct \
        --reuse
done

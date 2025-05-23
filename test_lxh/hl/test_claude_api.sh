export DOUBAO_VL_KEY='1648285843187896359'
export DOUBAO_VL_ENDPOINT='Doubao-1.5-vision-pro-32k'

eval "$('/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/mllm_env/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate mm-eureka-hl
echo "conda activate mm-eureka-hl"

EXP_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/lanxiaohan/mig_from_vacv/projects/mm_large_model/for_o1/VLMEvalKit_0509/VLMEvalKit
export PYTHONPATH=$PYTHONPATH:${EXP_DIR}

python /mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/lanxiaohan/mig_from_vacv/projects/mm_large_model/for_o1/VLMEvalKit_0509/VLMEvalKit/vlmeval/api/claude.py

export OPENAI_API_KEY=***
export OPENAI_API_BASE=***
export SiliconFlow_API_KEY=***

pkill -f -9 ray

declare -A gpu_mapping=(
    ["MathVista_MINI"]=0
    ["MathVerse_MINI_Vision_Only"]=1
    ["WeMath"]=0
    ["LogicVista"]=3
    ["MathVision"]=4
    ["DynaMath_noprompt"]=5
)
OUTPUT_DIR=./vlmeval_output
# echo "ckpt is $CKPT, step is ${STEP:-unset}"
export TOKENIZERS_PARALLELISM=false
for dataset in WeMath; do
# for dataset in MathVerse_MINI_Vision_Only MathVista_MINI WeMath LogicVista DynaMath; do
    echo "Processing dataset: $dataset"
    work_dir="${OUTPUT_DIR}/vlmeval_results_qwen25_sft/"
    mkdir -p ${work_dir}
    ckpt_path=xx/qwen25vl-mm-retool-sft-mixed-s30/
    work_dir=${work_dir}/qwen25_sft/
    mkdir -p ${work_dir}
    log_file=$work_dir/${dataset}_$(date +%Y%m%d_%H%M%S).log
    
    (   
        export TOKENIZERS_PARALLELISM=false
        export CUDA_VISIBLE_DEVICES=${gpu_mapping[$dataset]}
        python run.py --data "$dataset" \
            --work-dir "$work_dir" \
            --model Qwen2.5-VL-7B-Instruct-Eureka-CKPT \
            --ckpt "$ckpt_path" \
            --reuse \
            --judge 'qwen3-max' > "$log_file" 2>&1
    ) &
    echo "PID: $! -> $dataset"
done
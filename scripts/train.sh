#!/bin/bash
data_dir= #ADD HERE THE PATH TO THE DATASET

base_dir="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
echo base_dir $base_dir;
output_dir_base=$base_dir/training_output

declare -a model_arr=("CBRTiny")
declare -A freeze_arr=( ["efficientnet_b0"]=True ["CBRTiny"]=False )

declare -a classes_arr=(  "Cardiomegaly" ) #
declare -A class_to_index_arr=( ["Edema"]=5 ["Atelectasis"]=8 ["Cardiomegaly"]=2 ["Consolidation"]=6 ["Pleural Effusion"]=10)
declare -A upsample_arr=( ["Edema"]=True ["Atelectasis"]=True ["Cardiomegaly"]=True ["Consolidation"]=True ["Pleural Effusion"]=True)

frozen_model_to_load=best_model.pth
view=Frontal #Frontal Lateral
fp16=--fp16 # --fp16 or nothing for fp32
num_workers=12
in_memory= #--in-memory # --in-memory or nothing
use_cache=--use-cache # --use-cache or nothing

epochs_frozen=20
lr_frozen=0.0001
wd_frozen=0.000001


epochs_unfrozen=120
declare -a lr_arr=("0.0001")
declare -a wd_arr=( "10" "0.1" "0.01" "0.000001")
declare -a bs_arr=("256" "128" "64" "32")
declare -A eval_steps_arr=( ["256"]="250" ["128"]="500" ["64"]="1000")

early_stopping_patience=10
do_early_stopping_frozen="--do-early-stopping ${early_stopping_patience}" #or empty for not
do_early_stopping_unfrozen= #"--do-early-stopping ${early_stopping_patience}"# or empty for not

loss_weighting="--do-weight-loss-even"

max_steps="--max-steps -1" #-1 is full
max_dataloader_size=

DEBUG=False
if [ $DEBUG = True ]; then
    epochs_frozen=3
    epochs_unfrozen=3
    eval_steps=1
    max_steps="--max-steps 3"
    max_dataloader_size="--max-dataloader-size 1000"
    echo "max_steps $max_steps"
    echo "max_dataloader_size $max_dataloader_size"
fi

for model in ${model_arr[@]}
do
    for lr in ${lr_arr[@]}
    do
        lr_str=${lr/./_}
    for bs in ${bs_arr[@]}
    do       
            for wd in ${wd_arr[@]}
            do
            #https://stackoverflow.com/questions/9084257/bash-array-with-spaces-in-elements
            for ((i = 0; i < ${#classes_arr[@]}; i++))
            do
           
            class=${classes_arr[$i]}
            class_idx=${class_to_index_arr["$class"]}
            class_folder=${class/ /_}
            output_dir_frozen=$output_dir_base/frozen/${model}/${class_folder}/
            output_dir_unfrozen=$output_dir_base/unfrozen/${model}/${class_folder}/lr_${lr_str}/bs_${bs}/wd_${wd}/
           
            eval_steps=${eval_steps_arr["$bs"]}
              
            do_upsample=
            echo ${upsample_arr[$class]}
            if [ "${upsample_arr["$class"]}" = "True" ]; then
                do_upsample="--do-upsample"
            fi
           
            model_to_load_dir=
            if [ "${freeze_arr["$model"]}" = "True" ]; then
                model_to_load_dir=$output_dir_frozen/$frozen_model_to_load
                echo $model_to_load_dir;
                    if [ ! -f $model_to_load_dir  ]; then
                        echo "BASH TRAIN FROZEN model $model class $class class_idx $class_idx view $view lr $lr_frozen wd $wd_frozen batch_size $bs $do_upsample";
                        python train.py --model $model --class-positive $class_idx --view $view --data-dir $data_dir --output-dir $output_dir_frozen --num-epochs $epochs_frozen --lr $lr_frozen --wd $wd_frozen --batch-size $bs --do-train $max_steps $max_dataloader_size  $do_early_stopping_frozen --do-eval --eval-steps $eval_steps $loss_weighting $do_upsample --freeze $fp16 --num-workers $num_workers $in_memory $use_cache;
                    else
                        echo "BASH not training frozen found ${model_to_load_dir}";
                    fi
                    model_to_load_dir="--model-to-load-dir ${model_to_load_dir}"
            fi
            if [ ! -f $output_dir_unfrozen/last_model.pth ]; then
                echo "BASH UNFROZEN model $model class $class class_idx $class_idx view $view lr $lr wd $wd batch_size $bs";
                python train.py --model $model --class-positive $class_idx --view $view $model_to_load_dir --data-dir $data_dir --output-dir $output_dir_unfrozen --num-epochs $epochs_unfrozen --lr $lr --wd $wd --batch-size $bs --do-train $max_steps $max_dataloader_size $do_early_stopping_unfrozen --do-eval --eval-steps $eval_steps $loss_weighting $do_upsample $fp16 --num-workers $num_workers $in_memory $use_cache;
            else
                        echo "BASH not training unfrozen found ${output_dir_unfrozen}/last_model.pth";
            fi
                done       
        done
        done
    done
done

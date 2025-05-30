MASTER_PORT=$((RANDOM % 50001 + 10000))
forget_losses=(
    GA+GD
    GA+KL
    NPO+GD
    NPO+KL
    DPO+GD
    DPO+KL
    IDK+GD
    IDK+KL
    IDK+AP
    MPUT #MP-ME
    MPT  #MP-IDK
)
mask=true

# forget_losses=(  ## mask=false for ME+GD
#     ME+GD
# )
# mask=false 


task_list=(1)

# pass to python script
export TASK_LIST=$(IFS=,; echo "${task_list[*]}")

learning_rates=(
    1e-5
)
use_LoRA=false

if [ "$use_LoRA" = true ]; then
    save_root="results/tofu"
    num_epochs=(5 10)
    NODE=1
    DEVICE1=0
    DEVICE2=0
else
    save_root="results/tofu"
    num_epochs=(5 10)
    NODE=2
    DEVICE1="0,1"
    DEVICE2=0
fi

forget_coeff=1.0
regularization_coeff=1.0
save_checkpoint=true
save_steps=last
eval_steps=(last)


splits=(forget01 forget05 forget10)  
for split in ${splits[@]}; do
    for num_epoch in ${num_epochs[@]}; do
        for forget_loss in ${forget_losses[@]}; do
            for lr in ${learning_rates[@]}; do
                for task_id in ${task_list[@]}; do
                    COMMON="use_LoRA=$use_LoRA forget_coeff=$forget_coeff regularization_coeff=$regularization_coeff lr=$lr split=$split forget_loss=$forget_loss num_epochs=$num_epoch \
                        mask=$mask save_root=$save_root save_checkpoint=$save_checkpoint"
                    CUDA_VISIBLE_DEVICES=$DEVICE1 torchrun --nproc_per_node=$NODE --master_port=$MASTER_PORT \
                            forget.py \
                            --config-name=tofu.yaml \
                            task_id=$task_id \
                            save_steps=$save_steps \
                            $COMMON
                    for step in ${eval_steps[@]}; do
                        CUDA_VISIBLE_DEVICES=$DEVICE2 torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
                                eval.py \
                                --config-name=tofu.yaml \
                                task_id=$task_id \
                                eval_unlearn_step=$step \
                                $COMMON
                    done
                done
                CUDA_VISIBLE_DEVICES=$DEVICE2 python3 \
                eval_gpt.py \
                --lr $lr \
                --forget $split \
                --method $forget_loss \
                --batch_size 2 \
                --epochs $num_epoch \
                $([ "$use_LoRA" = "true" ] && echo --use_LoRA)
            done
        done
    done
done

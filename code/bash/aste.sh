while getopts ':d:s:c:t:n:l:z:e:D:' opt
do
    case $opt in
        d)
        dataset="$OPTARG" ;;
        s)
        seed="$OPTARG" ;;
        c)
        CUDA_IDS="$OPTARG" ;;
        t)
        table_encoder="$OPTARG" ;;
        n)
        num_table_layers="$OPTARG" ;;
        l)
        learning_rate="$OPTARG" ;;
        z)
        span_pruning="$OPTARG" ;;
        e)
        seq2mat="$OPTARG" ;;
        D)
        num_d="$OPTARG" ;;
        ?)
        exit 1;;
    esac
done


if [ ! "${table_encoder}" ]
then
  table_encoder=resnet
fi


if [ ! "${num_table_layers}" ]
then
  num_table_layers=2
fi


gradient_clip_val=1
warmup_steps=100
weight_decay=0.01
precision=16
batch_size=4
data_dir="../data/aste_data_bert/${dataset}"


CUDA_VISIBLE_DEVICES=${CUDA_IDS} python3 aste_train.py \
  --gpus=1 \
  --precision=${precision} \
  --data_dir ${data_dir} \
  --model_name_or_path 'bert-base-uncased' \
  --output_dir ../output/ASTE/${dataset}/ \
  --learning_rate ${learning_rate}e-5 \
  --train_batch_size ${batch_size} \
  --eval_batch_size ${batch_size} \
  --seed $seed \
  --warmup_steps ${warmup_steps} \
  --lr_scheduler linear \
  --gradient_clip_val ${gradient_clip_val} \
  --weight_decay ${weight_decay} \
  --max_seq_length -1 \
  --max_epochs 10 \
  --cuda_ids ${CUDA_IDS} \
  --do_train \
  --table_encoder ${table_encoder} \
  --num_table_layers ${num_table_layers} \
  --span_pruning ${span_pruning} \
  --seq2mat ${seq2mat} \
  --num_d ${num_d}
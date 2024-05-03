python train_tokencompressor.py --batch_size 32 --num_slots 7 --num_epochs 200 --max_samples 1000 --ablated_indices 1 3 5 --results_dir /shared/rzhang/slot_att/results/full-token-compressor --wandb --notes "ablated_indices_1_3_5"
# python train_tokencompressor.py --num_slots 7 --num_epochs 1000 --results_dir /shared/rzhang/slot_att/results/full-token-compressor --wandb
# python finetune.py --base_num_slots 7 --num_epochs 1000 --results_dir /shared/rzhang/slot_att/results/slots6-freeze7 --model_path="/shared/rzhang/slot_att/results/007-objects-all-slots-7.ckpt" --finetuned_model_name="slots6-freeze7"
# python finetune.py --base_is_finetuned True --base_num_slots 6 --num_epochs 1000 --results_dir /shared/rzhang/slot_att/results/slots5-freeze6 --model_path="/shared/rzhang/slot_att/results/slots6-freeze7/005-slots6-freeze7.ckpt" --finetuned_model_name="slots5-freeze6"
# GPU1 using
# python finetune.py --num_slots 6 --num_epochs 1000 --results_dir /shared/rzhang/slot_att/results/slots6-freeze7 --model_path="/shared/rzhang/slot_att/results/objects-all-slots-7.ckpt" --finetuned_model_name="slots6-freeze7" --learning_rate 1e-5
# GPU2 using
# python finetune.py --num_slots 6 --num_epochs 1000 --results_dir /shared/rzhang/slot_att/results/slots6-freeze7 --model_path="/shared/rzhang/slot_att/results/objects-all-slots-7.ckpt" --finetuned_model_name="slots6-freeze7"
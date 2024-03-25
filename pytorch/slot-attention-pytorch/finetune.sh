python finetune.py --num_slots 6 --num_epochs 1000 --results_dir /shared/rzhang/slot_att/results/slots6-freeze7 --model_path="/shared/rzhang/slot_att/results/007-objects-all-slots-7.ckpt" --finetuned_model_name="slots6-freeze7"
# GPU1 using
# python finetune.py --num_slots 6 --num_epochs 1000 --results_dir /shared/rzhang/slot_att/results/slots6-freeze7 --model_path="/shared/rzhang/slot_att/results/objects-all-slots-7.ckpt" --finetuned_model_name="slots6-freeze7" --learning_rate 1e-5
# GPU2 using
# python finetune.py --num_slots 6 --num_epochs 1000 --results_dir /shared/rzhang/slot_att/results/slots6-freeze7 --model_path="/shared/rzhang/slot_att/results/objects-all-slots-7.ckpt" --finetuned_model_name="slots6-freeze7"
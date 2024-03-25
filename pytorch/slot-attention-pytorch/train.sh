# python train.py --results_dir /shared/rzhang/slot_att/results --model_name="objects-all-slots-7" --dataset_name="CLEVR" 
# python train.py --results_dir /shared/rzhang/slot_att/results --model_name="objects-all-slots-7-L-encoder" --dataset_name="CLEVR" --cnn_depth 6 
# python train.py --results_dir /shared/rzhang/slot_att/results --model_name="objects-all-slots-7-trfmr" --dataset_name="CLEVR" --use_trfmr True
# python train.py --results_dir /shared/rzhang/slot_att/results --model_name="objects-all-slots-7-trfmr-full" --dataset_name="CLEVR" --use_trfmr True --use_trfmr_encoder True --use_trfmr_decoder True
python train.py --results_dir /shared/rzhang/slot_att/results --model_name="objects-all-slots-7-trfmr" --dataset_name="CLEVR" --use_trfmr True
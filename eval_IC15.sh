export PYTHONPATH=$PYTHONPATH:.
#TF_CUDNN_USE_AUTOTUNE=0
gpu=$1
python eval_KPN.py --gpu $gpu --score_kernel_th 0.2 --score_final_th 0.8 --test_size 1024 1920 --checkepoch 300 --store_img True --exp_name Icdar2015

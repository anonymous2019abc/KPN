export PYTHONPATH=$PYTHONPATH:.
#TF_CUDNN_USE_AUTOTUNE=0
gpu=$1
python eval_KPN.py --gpu $gpu --score_kernel_th 0.2 --score_final_th 0.6 --test_size 512 832 --checkepoch 830 --store_img True --exp_name Ctw1500

export PYTHONPATH=`pwd`

CUDA_VISIBLE_DEVICES="1" /home/dmlab/anaconda3/bin/python3 training_ptr_gen/train.py #>& log/training_log &


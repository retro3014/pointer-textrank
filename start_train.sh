export PYTHONPATH=`pwd`
CUDA_VISIBLE_DEVICES="1" python training_ptr_gen/train.py >& ../log/training_log &


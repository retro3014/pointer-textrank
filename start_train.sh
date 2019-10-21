export PYTHONPATH=`pwd`
<<<<<<< HEAD
CUDA_VISIBLE_DEVICES="1" /home/dmlab/anaconda3/bin/python3 training_ptr_gen/train.py #>& log/training_log &
=======
CUDA_VISIBLE_DEVICES="1"
python training_ptr_gen/train.py >& ../log/training_log &

>>>>>>> 01fa3dfa735cf6f96c96277830034e0437594938

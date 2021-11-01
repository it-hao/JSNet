####################################  TRAIN  ########################################
# x2
python main.py  --model BCAN --save BCAN_x2 --scale 2  --chop --save_results --patch_size 96

# x3
python main.py  --model BCAN --save BCAN_x3 --scale 3  --chop --save_results --patch_size 144

# x4
python main.py  --model BCAN --save BCAN_x4 --scale 4  --chop --save_results --patch_size 192

# My own setting
# x2
activate pytorch-gpu
python main.py --scale 2 --save_results --layers 4 --save_models --patch_size 96
python main.py --scale 2 --save_results --layers 6 --patch_size 96





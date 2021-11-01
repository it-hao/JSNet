# x2
python main.py --data_test MyImage --scale 2  --model BCAN --pre_train ../model/BCAN_x2.pt --test_only --save_results --chop --save BCAN  --testset Set5

python main.py --data_test MyImage --scale 2  --model BCAN --pre_train ../model/BCAN_x2.pt --test_only --save_results --chop  --self_ensemble --save BCAN  --testset Set5

# x3
python main.py --data_test MyImage --scale 3  --model BCAN --pre_train ../model/BCAN_x3.pt --test_only --save_results --chop --save BCAN  --testset Set5

python main.py --data_test MyImage --scale 3  --model BCAN --pre_train ../model/BCAN_x3.pt --test_only --save_results --chop  --self_ensemble --save BCAN  --testset Set5

# x4
python main.py --data_test MyImage --scale 4  --model BCAN --pre_train ../model/BCAN_x4.pt --test_only --save_results --chop --save BCAN  --testset Set5

python main.py --data_test MyImage --scale 4  --model BCAN --pre_train ../model/BCAN_x4.pt --test_only --save_results --chop  --self_ensemble --save BCAN  --testset Set5

# x8
python main.py --data_test MyImage --scale 8  --model BCAN --pre_train ../model/BCAN_x8.pt --test_only --save_results --chop --save BCAN  --testset Set5

python main.py --data_test MyImage --scale 8  --model BCAN --pre_train ../model/BCAN_x8.pt --test_only --save_results --chop  --self_ensemble --save BCAN  --testset Set5
source activate lwr
export PYTHONPATH='/home/zhao/liaowenrui/mycode7.0'
CUDA_VISIBLE_DEVICES=1 python create_results.py --create_results --testset BSD100
python calc_psnr_ssim.py --testset BSD100

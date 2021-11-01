import os
from skimage import io
from calc_psnr_ssim import calc_metrics
def main():
    img_dir = "./prepareimg"
    scale = 4
    SR_filenames = []
    hr_filename = "./prepareimg/78004_x4_HR.png"

    for entry in os.scandir(img_dir):
        # print(entry)
        filename = os.path.splitext(entry.name)[0]
        # print(filename)
        if 'SR' in filename:
            filename = os.path.join(img_dir, filename + '.png')
            SR_filenames.append(filename)
            SR_filenames.sort()
    print(SR_filenames, len(SR_filenames))
    psnr_log = open(img_dir + "/psnr_log" + '_x' + str(scale) + ".txt", 'a')
    for i, sr_filename in enumerate(SR_filenames):
        # 为了获得文件名
        print(sr_filename, hr_filename)
        (filepath, tempfilename) = os.path.split(sr_filename)
        (filename, extension) = os.path.splitext(tempfilename)
        hr_img = io.imread(hr_filename)
        sr_img = io.imread(sr_filename)
        psnr_val, ssim_val = calc_metrics(hr_img, sr_img, scale)  # 计算每个图片的psnr 和 ssim.
        # 计算psnr/ssim
        psnr_log.write(
            "{}:\tpsnr = {} ,\tssim = {}\n".format(filename.ljust(30, ' '), round(psnr_val, 3), round(ssim_val, 5)))
if __name__ == '__main__':
    main()
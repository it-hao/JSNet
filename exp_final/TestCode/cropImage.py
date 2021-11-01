import cv2
import os
def crop_images(name):

	crop_dir = "./crop"
	if not os.path.exists(crop_dir):
		os.makedirs(crop_dir)
	img = cv2.imread(name)
	imgcopy = img.copy() # 复制一份图片
	height, width, c = img.shape
	# print(height, width, c)
	crop_w, crop_h = 60, 40
	# (x1,y1)==>(x2,y2) 也就是长度为120，宽为60 的图片，这里自己选择
	x1 = 180
	y1 = 120
	x2 = x1 + crop_w
	y2 = y1 + crop_h

	# 抠矩形，颜色自己选择
	cv2.rectangle(imgcopy,(x1,y1),(x2,y2),(0,0,255),4) # BGR (0,0,255)红色 (0,255,0)绿色 (255,255,255)白色 (255,0,0)蓝色 0,255,255)黄色
	# crop = imgcopy[y1:y2+1, x1:x2+1] # 这里和下面一行代码功能一样，一个保存的裁剪的图片带颜色框，一个没有颜色框
	crop = img[y1:y2+1, x1:x2+1]
	crop.resize()
	imgcopy = imgcopy[0:(height//2), :]
	cv2.imwrite(name[:-4]+"2"+".png",imgcopy) # 保存带有框的图片
	cv2.imwrite(name[:-4]+"_crop"+".png",crop) # 保存切好的图片

if __name__ == '__main__':
	# bicubic VDSR FSRCNN LapSRN CARN IDN IMDN Ours
	dir = "./prepareimg"
	img = "78004"
	scale = 4
	methods = ["bicubic", "VDSR", "FSRCNN", "LapSRN", "CARN", "IDN", "IMDN", "RDAN", "HR"]
	for i, method in enumerate(methods):
		if method is "HR":
			image_name = dir +'/'+ img + "_x" + str(scale) + "_HR" + ".png"
		else:
			image_name = dir + '/' + img + "_x" + str(scale) + "_SR_" + method + ".png"
		print(image_name)
		crop_images(image_name)
	#img_078
	#x1 = 915 y1 = 89

# BI degradation model
# img_004
# x1 = 670
# y1 = 480
# x2 = 790
# y2 = 540

# YumeiroCooking
# x1 = 90
# y1 = 900
# x2 = 210
# y2 = 960

# img_016 
# x1 = 660
# y1 = 1060
# x2 = 780
# y2 = 1120	

# img_073
# x1 = 25
# y1 = 810
# x2 = 145
# y2 = 870	



# BD degradation model
#  img_024

# x1 = 580
# y1 = 120
# x2 = 700
# y2 = 180

# x1 = 600
# y1 = 370
# x2 = 720
# y2 = 430

# BD degradation model
# img_093

# x1 = 180
# y1 = 30
# x2 = 300
# y2 = 90
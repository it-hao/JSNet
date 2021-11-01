from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
#plt.figure(3)

x_index=np.arange(4)                        #柱的索引
x_data=("32×24","64×48","128×96","256×192")
y5_data=(59.1,63.2,65.3,74.1)
y4_data=(60.2,63.8,65.7,74.7)
y3_data=(50.4,52.4,55.2,66.9)
y2_data=(55.2,57.4,59,69.4)
y1_data=(57.0,59.8,62.8,71.5)
bar_width=0.25                              #定义一个数字代表柱的宽度

plt.plot(x_data,y4_data,lw=1.5,c='black',marker='o',ms=6,label='RSN')
plt.plot(x_data,y5_data,lw=1.5,c='orange',marker='v',ms=6,label='HRNet-W32')
plt.plot(x_data,y1_data,lw=1.5,c='red',marker='s',ms=6,label='MSPN')
plt.plot(x_data,y2_data,lw=1.5,c='blue', marker='*',ms=6,label='Simple-Baseline')
plt.plot(x_data,y3_data,lw=1.5,c='green',marker='^',ms=6,label='Hourglass')


plt.axis([-1,4,50,75])
plt.ylabel('COCO AP',fontsize=18,family='Times New Roman')
plt.xlabel('Bbox Size',fontsize=18,family='Times New Roman')
plt.xticks(x_index, x_data) #x轴刻度线
plt.tick_params(labelsize=12)
plt.grid(color='gray')
plt.legend(loc='center left',fontsize=11, bbox_to_anchor=(0.02, -0.28),ncol=3,frameon=False)
bwith = 0.5 #边框宽度设置为2
TK = plt.gca()#获取边框
TK.spines['bottom'].set_linewidth(bwith)#图框下边
TK.spines['left'].set_linewidth(bwith)#图框左边
TK.spines['top'].set_linewidth(bwith)#图框上边
TK.spines['right'].set_linewidth(bwith)#图框右边
#plt.legend(loc='SouthOutside',ncol=3)  #显示图例
plt.tight_layout()  #自动控制图像外部边缘，此方法不能够很好的控制图像间的间隔
plt.savefig(r'.\comparison.pdf',dpi=350,bbox_inches='tight')
plt.show()
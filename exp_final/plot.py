from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator
#plt.figure(3)
fig = plt.figure()
fig.set_facecolor('white')
x = [12, 665, 297, 677, 588, 1592, 703, 1014, 1019, 324, 440]
y = [28.53, 28.83, 28.95, 28.96, 29.03, 29.06, 29.09, 29.15, 29.09, 29.03, 29.11]
label = ['FSRCNN','VDSR','DRRN','MemNet','IDN','CARN','IMDN','ESRN','ESRN-F','ESRN-V','MRAN (Ours)']
# 设置figure窗体的颜色
plt.rcParams['figure.facecolor'] = 'white'
# 设置axes绘图区的颜色
plt.rcParams['axes.facecolor'] = 'white'
color = ["blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue", "red"]
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 12,
         }
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 10,
         }
for i in range(len(x)):
    plt.scatter(x[i], y[i], color=color[i])

for i in range(0, len(x)):
    plt.text(x[i], y[i], label[i], fontdict=font2, ha="center", va="bottom")
x_major_locator=MultipleLocator(500)
#把x轴的刻度间隔设置为1，并存在变量里
#ax为两条坐标轴的实例
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
plt.xlim(-120,2000)#把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白

# plt.legend(prop={'family' : 'Times New Roman', 'size': 16})
plt.gcf().set_facecolor(np.ones(3)* 240 / 255)   # 生成画布的大小
plt.grid()  # 生成网格

plt.legend(loc='center left',fontsize=11, bbox_to_anchor=(0.02, -0.28),ncol=3,frameon=False)

# 设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=10)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
for item in [fig, ax]:
    item.patch.set_visible(False)
plt.xlabel('Parameters (K)',font1)
plt.ylabel('PSNR (dB)',font1)
plt.show()
plt.tight_layout()
plt.savefig(".\\fig.pdf")
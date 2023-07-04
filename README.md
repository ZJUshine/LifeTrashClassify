# LifeTrashClassify

# 树莓派准备工作

## 烧录系统

1.下载[Raspberry Pi Imager](https://www.raspberrypi.com/software/) 使用[Raspberry Pi 4b烧录系统(官方烧录工具)](https://blog.csdn.net/weixin_64852743/article/details/127305952)
**烧写系统64位**
![image.png](https://zjushine-picgo.oss-cn-hangzhou.aliyuncs.com/img/20230628172456.png)
2.配置**用户名、密码、WIFI**等
默认账号：pi
默认密码：raspi
烧录完成后树莓派即可连接上WiFi，通过路由器或者手机查看等方法获取树莓派IP

## 连接树莓派

根据树莓派IP使用VNC连接
默认IP：172.20.10.5

## 更改软件源

使用vim编辑 /etc/apt/sources.list
更换清华源/浙大源/阿里源等
再使用 `sudo apt-get update` 更新

# 使用OpenCV拍摄图片

## 硬件连接

将CSI摄像头插入树莓派的接口 如下图所示
![image.png](https://zjushine-picgo.oss-cn-hangzhou.aliyuncs.com/img/20230704204615.png)

## 安装opencv

为了调用摄像头以及图像处理

```bash
python -m pip install opencv-python
```

## 拍摄代码

```python
import cv2
import numpy as np
name = 0
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
ret, frame = cap.read()
rows, cols, channels = frame.shape
# 图像预处理
def img_p(img):
    # 灰度化
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 平滑滤波
    blur = cv2.blur(gray_img, (3,3))
    # 二值化
    ret1, th1 = cv2.threshold(blur, 190, 255, cv2.THRESH_BINARY)
    # 透视变换
    b = 50
    pts1 = np.float32([[b, 0], [cols-b, 0], [0, rows], [cols, rows]])
    pts2 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(blur, M, (cols, rows))
    return dst
while(1):
        ret,frame = cap.read()
        dst = img_p(frame)
        cv2.imshow('usb camera', dst)
        k = cv2.waitKey(1)
        if (k == ord('q')):
            break
        elif(k == ord('s')):
                name += 1
                filename = str(name) + '.jpg'
                cv2.imwrite(filename, dst)
cap.release()
cv2.destroyAllWindows()
```

# 深度学习识别垃圾种类

## 数据集

中文生活垃圾分类数据集
147674张带中文标签的生活垃圾图像集，包含可回收垃圾、厨余垃圾、有害垃圾、其他垃圾4个标准垃圾大类，覆盖常见的食品，厨房用品，家具，家电等265个垃圾小类。其中训练集133038张图像，验证集14642张图像，均从海量中文互联网社区语料进行提取，整理出频率较高的常见生活垃圾名称，数据大小为13GB。
https://modelscope.cn/datasets/tany0699/garbage265/summary

## 补充数据集

使用Python爬虫爬取百度图片中的不同种类的垃圾
主函数如下：

```bash
python crawling.py --word "电池" --total_page 10 --start_page 1 --per_page 30
```

可以设置**抓取关键词**、**需要抓取的总页数**、**起始页数**、**每页大小**

## 现有模型调用方法

### Modelscope 

ModelScope是**阿里推出的下一代开源的模型即服务共享平台**，为泛AI开发者提供灵活、易用、低成本的一站式模型服务产品，其开发目标为让模型应用更简单。 ModelScope希望在汇集行业领先的预训练模型，减少开发者的重复研发成本。

### 安装环境

```bash
pip install modelscope
```

### 模型预测

```python
from modelscope.pipelines import pipeline 
from modelscope.utils.constant import Tasks 
img_path = '1.jpg' 
image_classification = pipeline(Tasks.image_classification, model='damo/cv_convnext-base_image-classification_garbage') 
result = image_classification(img_path) 
print(result)
```

img_path：需要被预测的图片路径
result：预测结果（输出四个概率最高的垃圾种类以及它们的概率）

## 自己训练模型

### 安装软件包

```bash
python -m pip install torch
python -m pip install torchvison
python -m pip install matplotlib
```

### 模型选取

MobileNetV3是一种轻量级的卷积神经网络模型，适用于移动设备和嵌入式设备上的计算机视觉任务。它使用了一种新型的网络结构和注意力机制，以及自动搜索方法，提高了模型的准确率和速度。

### 训练过程

在large的预训练模型上使用上述数据集训练30个epoch，最后正确率可以达到75%左右。

### 模型预测

模型结果可以输出top5正确率。

# 舵机控制垃圾桶开合

## 硬件连接

![image.png](https://zjushine-picgo.oss-cn-hangzhou.aliyuncs.com/img/20230704212551.png)

## 安装舵机控制包

```bash
pip3 install Adafruit_PCA9685
```

## 舵机控制代码

```python 
import time
# 导入Adafruit_PCA9685模块
import Adafruit_PCA9685

# 使用默认地址（0x40）初始化PCA9685。
pwm = Adafruit_PCA9685.PCA9685() 
# 或者指定不同的地址和/或总线：
# pwm = Adafruit_PCA9685.PCA9685(address=0x41, busnum=2)
def set_servo_angle(channel, angle):  # 输入角度转换成12^精度的数值
    date = int(4096 * ((angle * 11) + 500) / 20000 + 0.5) 
    pwm.set_pwm(channel, 0, date)

# 将频率设置为50赫兹，适合伺服系统。
pwm.set_pwm_freq(50)

print('Moving servo on channel x, press Ctrl-C to quit...')
# 选择需要移动的伺服电机的通道与角度
set_servo_angle(channel, angle)
time.sleep(1)
```

# 参考文献

[树莓派USB摄像头+python+opencv](https://blog.csdn.net/weever7/article/details/125782340)
[树莓派官方最新64位系统安装Pytorch和OpenCV](https://zhuanlan.zhihu.com/p/523875226)
[BaiduImageSpider](https://github.com/kong36088/BaiduImageSpider)
[树莓派：PCA9685控制伺服电机](https://www.jianshu.com/p/b95e5a90175a)
[MobileNetV3](https://github.com/yichaojie/MobileNetV3)

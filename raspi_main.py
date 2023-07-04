# 导入安装包
from modelscope.pipelines import pipeline 
from modelscope.utils.constant import Tasks 
import cv2
import numpy as np
import time
import Adafruit_PCA9685
# 初始化PCA9685

# 使用默认地址（0x40）初始化PCA9685。
pwm = Adafruit_PCA9685.PCA9685() 
# 或者指定不同的地址和/或总线：
# pwm = Adafruit_PCA9685.PCA9685(address=0x41, busnum=2)

# 将频率设置为50赫兹，适合伺服系统。
pwm.set_pwm_freq(50)

# 初始化摄像头
name = 0
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
ret, frame = cap.read()
rows, cols, channels = frame.shape

 
# 自定义函数

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

# 角度转换函数
def set_servo_angle(channel, angle):  # 输入角度转换成12^精度的数值
    date = int(4096 * ((angle * 11) + 500) / 20000 + 0.5) 
    pwm.set_pwm(channel, 0, date)

image_classification = pipeline(Tasks.image_classification, model='damo/cv_convnext-base_image-classification_garbage')
 
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
                results = image_classification(filename) 
                print(results)
                result = results['label'][0].split('-')[0]
                if (result=='有害垃圾'):
                    # 通道和角度
                    set_servo_angle(0, 90)
                if (result=='可回收垃圾'):
                    set_servo_angle(1, 90)
                if (result=='厨余垃圾'):
                    set_servo_angle(2, 90)
                if (result=='其他垃圾'):
                    set_servo_angle(3, 90)
                time.sleep(1)
cap.release()
cv2.destroyAllWindows()
import sys
sys.path.append('./model')
import torch
import torchvision.transforms as transforms
from model import MobileNetV3_large
from model import MobileNetV3_small
from PIL import Image
import torch.nn.functional as F
# 创建一个检测器类，包含了图片的读取，检测等方法
class Detector(object):
    # netkind为'large'或'small'可以选择加载MobileNetV3_large或MobileNetV3_small
    # 需要事先训练好对应网络的权重
    def __init__(self,net_kind,num_classes=17):
        super(Detector, self).__init__()
        kind=net_kind.lower()
        if kind=='large':
            self.net = MobileNetV3_large(num_classes=num_classes)
        elif kind=='small':
            self.net = MobileNetV3_large(num_classes=num_classes)
        self.net.eval()
        if torch.cuda.is_available():
            self.net.cuda()

    def load_weights(self,weight_path):
        self.net.load_state_dict(torch.load(weight_path))

    # 检测器主体
    def detect(self,weight_path,pic_path):
        # 先加载权重
        self.load_weights(weight_path=weight_path)
        # 读取图片
        img=Image.open(pic_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        img_tensor = transform(img).unsqueeze(0)
        if torch.cuda.is_available():
            img_tensor=img_tensor.cuda()
        net_output = self.net(img_tensor)
        # 使用 softmax 函数计算概率分布
        probs = F.softmax(net_output, dim=1)

        # 获取概率最高的 5 个标签
        top5_probs, top5_labels = torch.topk(probs, k=5)
        class_filename = 'classname.txt'
        f = open(class_filename, 'r')
        lines = f.readlines()
        for j in range(top5_probs.size(1)):
            print('预测结果 {}: {:.2f}%'.format(lines[top5_labels[0][j] - 1].strip(), top5_probs[0][j] * 100))


if __name__=='__main__':
    detector=Detector('large',num_classes=265)
    detector.detect('./weights/best.pkl','./3.png')








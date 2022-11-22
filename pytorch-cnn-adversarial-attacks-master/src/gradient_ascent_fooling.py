import os
import cv2
import numpy as np
import warnings

import torch
from torch.optim import SGD
from torchvision import models
from torch.nn import functional

from misc_functions import preprocess_image, recreate_image

warnings.filterwarnings("ignore", category=UserWarning)


class FoolingSampleGeneration:
    def __init__(self, model, target_class, minimum_confidence):
        # 输入模型，激活模型，设定攻击目标，目标置信度
        self.model = model
        self.model.eval()
        self.target_class = target_class
        self.minimum_confidence = minimum_confidence
        # 生成随机图像并显示，最小最大像素值，尺寸244*244*3
        self.created_image = np.uint8(np.random.uniform(0, 255, (224, 224, 3)))
        cv2.imshow('Random_image', self.created_image)
        cv2.waitKey()

    def generate(self):
        # 因为是目标攻击，因此此处设定了较大的攻击次数
        for i in range(1, 200):
            # 对图像进行预处理
            processed_image = preprocess_image(self.created_image)
            # 定义优化器，此处设置学习率
            optimizer = SGD([processed_image], lr=6)
            # 前向传播
            output = self.model(processed_image)
            # 由softmax层输出置信度
            a = torch.tensor([self.target_class])
            target_confidence = functional.softmax(output)[0][a].data[0]
            # 计算损失，输出i轮的置信度
            class_loss = -output[0, self.target_class]
            # 输出迭代轮数以及相应目标置信度
            print('Iteration:', str(i), 'Target Confidence', "{0:.4f}".format(target_confidence))
            if target_confidence > self.minimum_confidence:
                print('Generated fooling image with', "{0:.2f}".format(target_confidence),
                      'confidence at', str(i) + 'th iteration.')
                cv2.imwrite('../generated/ga_fooling_class_' + str(self.target_class) + '.jpg',
                            self.created_image)
                break
            # 梯度清零
            self.model.zero_grad()
            # 反向传播
            class_loss.backward()
            # 更新图像
            optimizer.step()
            # 生成图像 / 还原图像
            self.created_image = recreate_image(processed_image)
        return 1


if __name__ == '__main__':
    #  这里的483表示攻击目标的ID
    target_class = 483  # Castle
    pretrained_model = models.alexnet(pretrained=True)
    #  输入攻击模型，目标攻击类别，目标最小置信度，构造cig对象
    cig = FoolingSampleGeneration(pretrained_model, target_class, 0.99)
    #  运行生成对抗样本
    cig.generate()

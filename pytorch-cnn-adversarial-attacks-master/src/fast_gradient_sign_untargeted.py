import os
import numpy as np
import cv2
import warnings
import torch
from torch import nn
import torch.nn.functional as F

from misc_functions import preprocess_image, recreate_image, get_params

warnings.filterwarnings("ignore", category=UserWarning)


class FastGradientSignUntargeted:
    def __init__(self, model, alpha):
        # 输入模型，激活模型，设置学习率
        self.model = model
        self.model.eval()
        self.alpha = alpha

    #  输入原始图像和原始图像的ID
    def generate(self, original_image, im_label):
        # 转换成tensor张量便于计算
        im_label_as_var = torch.tensor([1])
        im_label_as_var[0] = im_label
        # 定义损失函数
        ce_loss = nn.CrossEntropyLoss()
        # 图像预处理
        processed_image = preprocess_image(original_image)
        # 开始迭代，尝试在10次迭代中成功攻击
        for i in range(10):
            print('Iteration:', str(i + 1))
            # 梯度清零
            processed_image.grad = None
            # 前向传播
            out = self.model(processed_image)
            # 计算损失模型损失
            pred_loss = ce_loss(out, im_label_as_var)
            # 反向传播
            pred_loss.backward()
            # 生成噪声，全局扰动
            adv_noise = self.alpha * torch.sign(processed_image.grad.data)
            # 将对抗扰动添加到预处理后的图像中
            processed_image.data = processed_image.data + adv_noise
            # 生成合成图像
            recreated_image = recreate_image(processed_image)
            # 对合成图像进进行处理
            prep_confirmation_image = preprocess_image(recreated_image)
            # 确认图像的前向传播
            confirmation_out = self.model(prep_confirmation_image)
            # 获取预测下标和置信度
            _, confirmation_prediction = confirmation_out.data.max(1)
            confirmation_confidence = nn.functional.softmax(confirmation_out)[0][confirmation_prediction].data[0]
            # 判断是否成功攻击，且是否达到预期的置信度
            if confirmation_prediction != im_label and confirmation_confidence >= 0.99:
                print('Original image was predicted as:    ', im_label)
                print('With adversarial noise converted to:', confirmation_prediction.item())
                print('And predicted with confidence of:   ', confirmation_confidence.item())
                noise_image = recreated_image - original_image
                # 写入噪声图像和对抗样本图像
                cv2.imwrite('../generated/untargeted_adv_noise_from_' + str(im_label) +
                            '_to_' + str(confirmation_prediction.item()) + '.jpg', noise_image)
                cv2.imwrite('../generated/untargeted_adv_img_from_' + str(im_label) +
                            '_to_' + str(confirmation_prediction.item()) + '.jpg', recreated_image)
                break
        return 1


if __name__ == '__main__':
    #  2表示从列表中取出的样本序号，这里代表的是bird，是被攻击的目标ID
    target_example = 2  # Eel
    #  调用获取参数函数，进行变量赋值
    (original_image, prep_img, target_class, _, pretrained_model) = get_params(target_example)
    #  输入AlexNet模型，学习率α实例化对象
    FGS_untargeted = FastGradientSignUntargeted(pretrained_model, 0.01)
    #  调用对象成员方法，用到了original_image, target_class = 13（鸟）两个变量
    FGS_untargeted.generate(original_image, target_class)
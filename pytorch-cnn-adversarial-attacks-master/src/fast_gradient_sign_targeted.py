import os
import numpy as np
import cv2
import warnings
import torch
from torch import nn
from misc_functions import preprocess_image, recreate_image, get_params

warnings.filterwarnings("ignore", category=UserWarning)


class FastGradientSignTargeted:
    def __init__(self, model, alpha):
        # 输入模型，激活模型，设置学习率
        self.model = model
        self.model.eval()
        self.alpha = alpha

    def generate(self, original_image, org_class, target_class):
        # 转换成tensor张量时因为输入交叉熵损失时要求为张量
        im_label_as_var = torch.tensor([1])
        im_label_as_var[0] = target_class
        # 定义损失函数
        ce_loss = nn.CrossEntropyLoss()
        # 图像预处理
        processed_image = preprocess_image(original_image)
        # 开始迭代，尝试在10轮中实现攻击
        for i in range(100):
            print('Iteration:', str(i + 1))
            # 对参数的梯度清零
            processed_image.grad = None
            # 前向传播
            out = self.model(processed_image)
            # 计算交叉熵损失
            pred_loss = ce_loss(out, im_label_as_var)
            # 反向传播
            pred_loss.backward()
            # 生成噪声，全局扰动
            adv_noise = self.alpha * torch.sign(processed_image.grad.data)
            # 将对抗扰动“添加”到预处理后的图像中，此处是“-”，即更新参数
            processed_image.data = processed_image.data - adv_noise
            # 生成合成图像
            recreated_image = recreate_image(processed_image)
            # 对合成图像进进行处理
            prep_confirmation_image = preprocess_image(recreated_image)
            # 确认图像的前向传播
            confirmation_out = self.model(prep_confirmation_image)
            # 获取预测下标和置信度
            _, confirmation_prediction = confirmation_out.data.max(1)
            confirmation_confidence = nn.functional.softmax(confirmation_out)[0][confirmation_prediction].data[0]
            # 检查目标攻击是否成功，且是否达到预期置信度
            if confirmation_prediction == target_class:
                print('Original image was predicted as:', org_class,
                      'with adversarial noise converted to:', confirmation_prediction.item(),
                      'and predicted with confidence of:', confirmation_confidence.item())
                noise_image = original_image - recreated_image
                # 写入噪声图像和对抗样本图像
                cv2.imwrite('../generated/targeted_adv_noise_from_' + str(org_class) +
                            '_to_' + str(confirmation_prediction.item()) + '.jpg', noise_image)
                cv2.imwrite('../generated/targeted_adv_img_from_' + str(org_class) +
                            '_to_' + str(confirmation_prediction.item()) + '.jpg', recreated_image)
                break
            else:
                print('Current label:', confirmation_prediction.item())
                print('Current confidence:', confirmation_confidence.item())
                print('Processed image Sum:', processed_image.data.sum().item())
        return 1


if __name__ == '__main__':
    # 原始图像只有数组中的几张，但攻击的目标可以是多种的
    target_example = 0  # Apple
    (original_image, prep_img, org_class, _, pretrained_model) = get_params(target_example)
    target_class = 62  # Mud turtle

    FGS_targeted = FastGradientSignTargeted(pretrained_model, 0.01)
    FGS_targeted.generate(original_image, org_class, target_class)

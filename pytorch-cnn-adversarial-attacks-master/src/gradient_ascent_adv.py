import warnings

import cv2
from torch.nn import functional
from torch.optim import SGD

from misc_functions import preprocess_image, recreate_image, get_params

warnings.filterwarnings("ignore", category=UserWarning)


class DisguisedFoolingSampleGeneration:
    def __init__(self, model, initial_image, target_class, minimum_confidence):
        self.model = model
        self.model.eval()
        self.target_class = target_class
        self.minimum_confidence = minimum_confidence
        self.initial_image = initial_image

    def generate(self):
        # 这里针对原图像进行的目标攻击，学习率小很多，收敛较慢，因此设置了相对更大的攻击轮数
        for i in range(1, 500):
            # 对图像进行预处理
            processed_image = preprocess_image(self.initial_image)
            # 定义优化器，此处的学习率设置为0.7，相比另一个方法小很多
            optimizer = SGD([processed_image], lr=0.7)
            # 前向传播
            output = self.model(processed_image)
            # 由softmax层输出置信度
            target_confidence = functional.softmax(output)[0][self.target_class].data
            # 计算损失，输出i轮的置信度
            class_loss = -output[0, self.target_class]
            # 输出迭代轮数以及相应目标置信度
            print('Iteration:', str(i), 'Target confidence', "{0:.4f}".format(target_confidence))
            if target_confidence > self.minimum_confidence:
                print('Generated disguised fooling image with', "{0:.2f}".format(target_confidence),
                      'confidence at', str(i) + 'th iteration.')
                cv2.imwrite('../generated/ga_adv_class_' + str(self.target_class) + '.jpg',
                            self.initial_image)
                break
            # 梯度清零
            self.model.zero_grad()
            # 反向传播
            class_loss.backward()
            # 更新图像
            optimizer.step()
            # 生成图像 / 还原图像
            self.initial_image = recreate_image(processed_image)
        return 1


if __name__ == '__main__':
    #  0代表样本序号，这里的0表示Apple，ID为948
    target_example = 0  # Apple
    (original_image, prep_img, _, _, pretrained_model) = get_params(target_example)
    fooling_target_class = 398  # Abacus
    min_confidence = 0.99
    #  输入攻击模型，原始图像，目标攻击类别，目标最小置信度，构造cig对象
    fool = DisguisedFoolingSampleGeneration(pretrained_model, original_image, fooling_target_class, min_confidence)
    #  运行生成对抗样本
    fool.generate()
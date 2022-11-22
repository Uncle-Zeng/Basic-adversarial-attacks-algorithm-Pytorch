import copy
import cv2
import numpy as np

import torch
from torch.autograd import Variable
from torchvision import models


#  输入CNNs的图像预处理函数，方便进行计算操作
def preprocess_image(cv2im, resize_im=True):
    # mean and std list for channels (Imagenet)  ImageNet图像通道的均值和标准差
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image  改变图片大小，压缩成了224×224
    if resize_im:
        cv2im = cv2.resize(cv2im, (224, 224))
    im_as_arr = np.float32(cv2im)  # 转换数据类型为float32
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels  标准化通道数据
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor  变换到float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable  转换成pytorch变量
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


#  从torch变量重新创建图像，类似于反向预处理，图像还原；
def recreate_image(im_as_var):
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1 / 0.229, 1 / 0.224, 1 / 0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    # Convert RBG to GBR
    recreated_im = recreated_im[..., ::-1]
    return recreated_im


#  输入列表序号，获取图片的相关信息，以及对图片进行预处理；
def get_params(example_index):

    # Pick one of the examples  样本列表，图片路径与ID相对应的二维数组
    example_list = [['../input_images/apple.JPEG', 948],
                    ['../input_images/eel.JPEG', 390],
                    ['../input_images/bird.JPEG', 13]]
    #  这里的example_index就是输入函数的target_example的值
    selected_example = example_index
    #  两行代码，分别取出选择样本的路径和ID
    img_path = example_list[selected_example][0]
    target_class = example_list[selected_example][1]
    #  rfind函数用法确定出区间，后取出中间段的字符串
    file_name_to_export = img_path[img_path.rfind('/') + 1:img_path.rfind('.')]
    # Read image  读取原始图像
    original_image = cv2.imread(img_path, 1)
    # Process image  图像预处理
    prep_img = preprocess_image(original_image)
    # Define model  定义模型
    pretrained_model = models.alexnet(pretrained=True)
    return (original_image,
            prep_img,
            target_class,
            file_name_to_export,
            pretrained_model)
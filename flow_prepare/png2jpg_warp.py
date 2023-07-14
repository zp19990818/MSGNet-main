# 将png转jpg与frame进行wrap
# 2022-11-03

import os
import time

import cv2
import numpy as np
from PIL import Image
from matplotlib import image as mpimg


def gray2rgb(arr, color_dict):
    """
    convert gray image into RGB image
    :param gray: single channel image with numpy type
    :param color_dict: color map
    :return:  rgb image
    """
    zeros1 = arr.copy()
    zeros2 = arr.copy()

    np.place(arr, arr == 1, 255)
    np.place(zeros1, zeros1 == 1, 0)
    np.place(zeros2, zeros2 == 1, 255)
    # print(zeros2)
    c = np.dstack((arr, zeros1, zeros2))

    return c


if __name__ == '__main__':

    time_begin = time.time()
    path = "../flow_prepare/example/"
    list = os.listdir(path)

    value = -0.25  # 范围-1至1
    basedOnCurrentValue = True  # 0或者1

    for i in range(len(list)):
        image_path = path + list[i] + "\\frame\\"
        mask_path = path + list[i] + "\\FCN_pred\\"
        save_path = path + list[i] + "\\wrap\\"
        list2 = os.listdir(image_path)
        list3 = os.listdir(mask_path)

        ## 处理png2jpg并保存不同路径
        for j in range(len(list2)):
            mask = mpimg.imread(mask_path + list3[j])
            mask = gray2rgb(mask, [255, 0, 255])  # 修改mask为紫色，速度较慢

            # frame进行双边滤波
            image = cv2.imread(image_path + list2[j])
            filter_image = cv2.bilateralFilter(src=image, d=0, sigmaColor=30, sigmaSpace=5)

            mask = mask.astype(float)
            filter_image = filter_image.astype(float)
            # wrap
            wrap_img = cv2.addWeighted(filter_image, 0.75, mask, 1, 0)

            if not os.path.exists(save_path):  # os模块判断并创建
                os.mkdir(save_path)
                cv2.imwrite(save_path + list2[j], wrap_img)
            else:
                cv2.imwrite(save_path + list2[j], wrap_img)

            print("wrap image: " + list2[j])

time_end = time.time()
time = time_end - time_begin
print('time:', time)

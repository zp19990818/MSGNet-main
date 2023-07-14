# 将不同目录下的测试结果转移到一个目录里

# coding=utf-8
import os
import shutil

#目标文件夹，此处为相对路径，也可以改为绝对路径
save_path = 'D:\\zhangpeng\\git_projects\\MSGNet-main\\result\\MSGNet\\MOCA\\MSGNet\\all\\'
if not os.path.exists(save_path):
    os.makedirs(save_path)

#源文件夹路径
path = "D:\\zhangpeng\\git_projects\\MSGNet-main\\result\\MSGNet\\MOCA\\MSGNet\\result\\"
folders = os.listdir(path)
for folder in folders:
    dir = path + '\\' + str(folder)
    files = os.listdir(dir)
    for file in files:
        source = dir + '\\' + str(file)
        deter = save_path + str(file)
        shutil.copyfile(source, deter)

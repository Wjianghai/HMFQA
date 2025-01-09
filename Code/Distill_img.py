# coding=gbk
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from torchvision.models.resnet import resnet50, resnet18
import cv2, torch, json, time, imageio, os, gc, collections, re, ast, filetype, imghdr
from transformers import logging
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch._six import string_classes
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import torch.backends.cudnn as cudnn
from pathlib import Path
from torch.profiler import profile, record_function, ProfilerActivity

# 常量定义
np_str_obj_array_pattern = re.compile(r'[SaUO]')
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")
logging.set_verbosity_error()
root = '/raid/wjh'
recipe_image_path = '/20210702Creeper/creeper/images'
MKG_path = '/20211029FoodVQA/Model/MKGEmbedding/Feature_ext/MKG'
feature_file_old = '/20211029FoodVQA/Model/MKGEmbedding/Feature_ext/PreMKG'
feature_file = '/20211029FoodVQA/Model/MKGEmbedding/Feature_ext/PreMKG_pth'
process_image_path = MKG_path + '/precess_image/precess_image'

device = 6
batch_size = 2
recipes_csv = pd.read_csv(
    r'/home/a100test1/mmfoodkg/wjh/20211029FoodVQA/Model/MKGEmbedding/Feature_ext/MKG/Entity/recipe_list.csv').values.tolist()
dataset_recipe = torch.load(r"/home/a100test1/mmfoodkg/mydir/KG/ChineseCLIP/recipe_id2tensor1130.pth",
                            map_location=lambda storage, loc: storage.cuda(6))

def json_read(file):
    with open(file, 'r') as f:
        ans = json.load(f)
    return ans


def json_write(file):
    with open(file, 'w') as f:
        json.dump(p, f)
    return 0


def getprocesspath(id):
    dirs = int(id) // 1000
    return "{}/{}".format(dirs, id)


path0 = r"/home/a100test1/mmfoodkg/mydir/pokemon.jpeg"


def image_tensor(path):
    image = cv2.imread("{}".format(path))
    if image is None:
        print(path)
        image = cv2.imread("{}".format(path0))

    size = (400, 400)
    shrink_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)  # 图片（输入端长度统一处理 ）放缩
    image_tensor = torch.from_numpy(shrink_image).unsqueeze(0).transpose(1, 3).float()
    return image_tensor


def gettensor(id, batch_data):
    global nopath
    global gifimgnum
    img_emb_re = []
    img_emb_cl = []

    path = '/home/a100test1/mmfoodkg/wjh/20210702Creeper/creeper/images/{}.jpg'.format(getprocesspath(id))
    try:
        img = image_tensor(path)
        img_emb_cl.append(batch_data['image'])
        img_emb_re.append(img.to(device))
    except:
        print(path)

    for num, item in enumerate(batch_data['process']):
        temppath = "/home/a100test1/mmfoodkg/wjh/20211029FoodVQA/Model/MKGEmbedding/Feature_ext/MKG/process_image_ciisr/{}_{}.jpg".format(
            getprocesspath(id), str(num))
        if Path(temppath).is_file():
            if imghdr.what(temppath):
                if filetype.guess(temppath).extension != "gif":
                    img_emb_cl.append(item['picture'])
                    img = image_tensor(temppath)
                    img_emb_re.append(img.to(device))
                else:
                    gifimgnum += 1
        else:
            gifimgnum += 1

    return img_emb_re, img_emb_cl


class Distillation_(torch.nn.Module):
    def __init__(self, resnet_picture_dim=1000, chnclip_picture_dim=1024, device="cuda:6"):
        super(Distillation_, self).__init__()
        self.device = device
        self.model0 = resnet18(weights=True)
        self.picture_linear = nn.Linear(resnet_picture_dim, chnclip_picture_dim, bias=True)  # 1000,1024,
        self.picture_loss_fn = nn.MSELoss()

    def forward(self, batch_resnet_picture, batch_chnclip_picture):
        batch_resnet_picture, batch_chnclip_picture = batch_resnet_picture, batch_chnclip_picture
        image_resnet_tensor = self.model0(batch_resnet_picture.squeeze(1))
        image_resnet_tensor = self.picture_linear(image_resnet_tensor)
        picture_loss = self.picture_loss_fn(image_resnet_tensor, batch_chnclip_picture)
        return picture_loss


if __name__ == '__main__':

    model = Distillation_()  # 加载模型
    model.to(device)
    for epoch in range(100):
        nopath = 0
        gifimgnum = 0
        pic_loss = 0
        for id, batch_data in enumerate(dataset_recipe):  # 取batch_size个 切片，
            path = r'/home/a100test1/mmfoodkg/wjh/20210702Creeper/creeper/images/{}.jpg'.format(getprocesspath(id))
            if Path(path).is_file():
                if imghdr.what(path):
                    batch_resnet_picture, batch_chnclip_picture = gettensor(id, batch_data)
                    picture_loss = model(
                        torch.tensor([item.cpu().detach().numpy() for item in batch_resnet_picture]).cuda(6),
                        torch.tensor([item.cpu().detach().numpy() for item in batch_chnclip_picture]).cuda(6))
                    picture_loss0 = picture_loss.item() / len(batch_resnet_picture)
                    pic_loss += picture_loss0
                    optimizer = optim.SGD(model.parameters(), lr=1e-2)
                    torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.68)
                    optimizer.zero_grad()  # 为这个训练步骤清除梯度
                    picture_loss.backward()  # 反向传播，计算梯度
                    optimizer.step()  # 优化步骤
            else:
                nopath += 1
            print("id is:", id)
            if (id % 100 != 0):
                print(picture_loss0, picture_loss.item())  # 累加平均
        print("epoch is:", epoch, nopath, gifimgnum)
        torch.save(model.state_dict(), 'img_emb_{}.pt'.format(epoch))





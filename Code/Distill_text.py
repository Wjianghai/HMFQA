# coding=gbk
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from torchvision.models.resnet import resnet50, resnet18
import cv2, torch, json, time, imageio, os, gc, collections, re, ast, numpy
from transformers import logging
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch._six import string_classes
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import torch.backends.cudnn as cudnn
from pathlib import Path

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

device = 6  # 3

recipes_csv = pd.read_csv(
    r'/home/a100test1/mmfoodkg/wjh/20211029FoodVQA/Model/MKGEmbedding/Feature_ext/MKG/Entity/recipe_list.csv').values.tolist()
dataset_recipe = torch.load(r"/home/a100test1/mmfoodkg/mydir/KG/ChineseCLIP/recipe_id2tensor1130.pth",
                            map_location=lambda storage, loc: storage.cuda(device))
tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')


def gettensor(id, batch_data):
    """
    返回：img_emb_re,img_emb_cl
    img_emb_re 是文本列表
    img_emb_cl 是已经提好的特征
    """
    text_encoded_input_re = []
    text_emb_cl = []
    text_emb_cl.append(batch_data['name'])

    meta_data = recipes_csv[id]
    meta_data_name = meta_data[2]
    encoded_input = tokenizer(meta_data_name, return_tensors='pt', padding=True)
    text_encoded_input_re.append(encoded_input)

    meta_pro = ast.literal_eval(meta_data[8])
    for item, item_meta in zip(batch_data['process'], meta_pro):
        try:
            item_meta_text = item_meta["text"]
            encoded_input = tokenizer(item_meta_text, return_tensors='pt', padding=True)
            text_encoded_input_re.append(encoded_input)
            text_emb_cl.append(item['text'])
        except:
            print(meta_pro)

    return text_encoded_input_re, text_emb_cl


class Distillation_text(torch.nn.Module):
    def __init__(self, test_tensor_dim=768, chnclip_text_dim=1024, device=device):
        super(Distillation_text, self).__init__()
        self.device = device
        self.Roberta = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext')
        self.text_linear = nn.Linear(in_features=test_tensor_dim, out_features=chnclip_text_dim,
                                     bias=True)  # 1000,1024,
        self.text_loss_fn = nn.MSELoss()

    def forward(self, batch_encoded_input, batch_chnclip_text):
        roberta_output = self.Roberta(**batch_encoded_input)  # （batch，token_size，768）
        roberta_output = roberta_output["pooler_output"]  # [1024]
        roberta_output, batch_chnclip_text = roberta_output, batch_chnclip_text

        test_tensor = self.text_linear(roberta_output.squeeze(0))
        test_tensor, batch_chnclip_text = test_tensor, batch_chnclip_text
        text_loss = self.text_loss_fn(test_tensor, batch_chnclip_text)

        return text_loss  # 【batch,candidate_num】


if __name__ == '__main__':

    model = Distillation_text()  # 加载模型
    model.to(device)
    for epoch in range(10):
        text_loss = 0
        for id, batch_data in enumerate(dataset_recipe):
            text_encoded_input_re, batch_chnclip_text = gettensor(id, batch_data)
            for bert_text, chnclip_text in zip(text_encoded_input_re, batch_chnclip_text):
                bert_text, chnclip_text = bert_text, chnclip_text
                text_loss = model(bert_text.to(device), chnclip_text.to(device))
                text_loss0 = text_loss.item() / len(text_encoded_input_re)
                text_loss += text_loss0
                optimizer = optim.SGD(model.parameters(), lr=1e-3)
                torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.68)
                optimizer.zero_grad()
                text_loss.backward()
                optimizer.step()
            print(id, text_loss0, text_loss.item())  # 累加平均
            if (id % 10000 == 9999):
                torch.save(model.state_dict(), 'text_emb_{}_{}_1211_12.pt'.format(epoch, id + 1))  # 累加平均
        print("epoch is:", epoch)
        torch.save(model.state_dict(), 'text_emb_{}_1211_12.pt'.format(epoch))

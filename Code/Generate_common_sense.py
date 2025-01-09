import torch,os, csv,json,requests,time
import concurrent.futures
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

lst = ["减肥", "高血糖", "高血脂", "高血压"]
file = open("recipe.txt", 'r')
recipelist = []

for item in file:
    item = item.strip().split('\n')
    recipelist.append(item)
file.close()

tokenizer = AutoTokenizer.from_pretrained("/home/a100test1/mmfoodkg/mydir/ChatGLM-6B", trust_remote_code=True)
model = AutoModel.from_pretrained("/home/a100test1/mmfoodkg/mydir/ChatGLM-6B", trust_remote_code=True).half().cuda()
tokenizer1 = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')


def find_all_elements(s: str, lst: list) -> dict:
    """
    判断字符串s中是否包含列表lst中的元素
    """
    result = []
    for elem in lst:
        indices = []
        idx = s.find(elem)
        while idx != -1:
            indices.append(idx)
            idx = s.find(elem, idx + 1)
        if indices:
            result.append(elem)
    return result

class Mytextmodel(torch.nn.Module):
    def __init__(self, test_tensor_dim=768, chnclip_text_dim=1024):
        super(Mytextmodel, self).__init__()
        self.Roberta = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext')
        self.text_linear = nn.Linear(in_features=test_tensor_dim, out_features=chnclip_text_dim, bias=True)

    def forward(self, batch_encoded_input):
        roberta_output = self.Roberta(**batch_encoded_input)
        test_tensor = roberta_output["pooler_output"]
        # test_tensor = self.text_linear(test_tensor)
        return test_tensor


def rob(text_input):
    text_input = tokenizer1(text_input, return_tensors='pt', padding=True)
    text_embedding = mytextmodel(text_input)
    return text_embedding.detach()

def getchatgpt(str):
    response, history = model.chat(tokenizer, str, history=[])
    return response

mytextmodel = Mytextmodel()
mytextmodel.load_state_dict(torch.load(r'/home/a100test1/mmfoodkg/mydir/Distillation/chn/dismodelcode/text_emb_1_120000_1211_12.pt',map_location='cuda:0'))

for filename in ["dataset_test", "dataset_10K"]:
    with open(r"/home/a100test1/mmfoodkg/mydir/chatmodels/commonsense_{}0507.csv".format(filename), mode='a', newline='', encoding="UTF-8") as csv_file:
        dataset = torch.load( r'/home/a100test1/mmfoodkg/wjh/20211029FoodVQA/Model/MKGVQA/DatasetProcess/dataset_file/{}'.format(filename), map_location='cuda:0')
        list = []
        list_time = []

        for item in tqdm(dataset):
            question_text = item['question_text']
            health_tag = find_all_elements(question_text, lst)  # 可能为1可能为2
            recipename = recipelist[(int(item['triple'][0][0][0]))]
            if len(health_tag) == 1:
                prompt = "请归纳{}和{}的关系".format(recipename[0], health_tag[0])
            elif len(health_tag) == 2:
                prompt = "请归纳{}和{}和{}的关系".format(recipename[0], health_tag[0], health_tag[1])

            new_question_text = "常识:{}问题：{}".format(getchatgpt(prompt).strip().replace(' ', '').replace('\n', '')[:400],question_text)


            writer = csv.writer(csv_file)
            writer.writerow([item['triple'],new_question_text])
            newqebeding = rob(new_question_text)

            item['question_text'] = new_question_text
            item['question_tensor'] = newqebeding
            list.append(item)
    torch.save(np.array(list), r'/home/a100test1/mmfoodkg/mydir/chatmodels/{}_0506_chatgpt_commonsense.pth'.format(filename))
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import collections
from torch._six import string_classes
import re
import torch
import os

abs_file = os.path.abspath(__file__)
abs_dir = abs_file[:abs_file.rfind('\\')] if os.name == 'nt' else abs_file[:abs_file.rfind(r'/')]
np_str_obj_array_pattern = re.compile(r'[SaUO]')
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


class MKG_Mul:
    def __init__(self,device = "cuda:0"):
        print("loading multimodal knowledge graph......")
        recipe_id2tensor_path = os.path.join(abs_dir, 'MFKG/KG_pth/recipe_id2tensor_768')
        self.recipe_id2tensor = torch.load(  # 7.3GB
            recipe_id2tensor_path,
            map_location={'cpu': "cpu"}
        )
        torch.save(  # 7.3GB
            self.recipe_id2tensor,
            "recipe_id2tensor_768"
        )
        self.device = device

    def context_gen(self,batch_data):
        ans = batch_data["ans"]
        ans_context = []
        for one_batch in ans:  # 一个ans100个batch
            ans_context_one_batch = []
            for one_triple_list in one_batch:  # 一个batch100个triple_list
                # 把头实体提取出来
                recipe_list = self.get_recipe(one_triple_list)
                # 把相关信息构造成字典输出
                ans_context_one_batch.append(self.get_context(recipe_list))
            ans_context.append(ans_context_one_batch)
        return ans_context
    def context_gen_test(self,batch_data):
        ans = batch_data["test_ans"]
        ans_context = []
        for one_batch in ans:  # 一个ans100个batch
            ans_context_one_batch = []
            for one_triple_list in one_batch:  # 一个batch100个triple_list
                # 把头实体提取出来
                recipe_list = self.get_recipe(one_triple_list)
                # 把相关信息构造成字典输出
                ans_context_one_batch.append(self.get_context(recipe_list))
            ans_context.append(ans_context_one_batch)
        return ans_context
    def get_recipe(self,one_triple_list):
        recipe_list = []
        for triple in one_triple_list:
            if triple[0] not in recipe_list:
                recipe_list.append(triple[0])
        return recipe_list
    def get_context(self,batch_recipe_id):
        # 用list做不定长特征的收集
        ans = {}
        # 收集所有的图片和文本
        text = []
        image = []
        for recipe_id in batch_recipe_id:
            nutrition_analysis = self.recipe_id2tensor[recipe_id]["nutrition_analysis"].unsqueeze(0) # 768
            step_list = []
            cur_process = self.recipe_id2tensor[recipe_id]["process"]
            for step in cur_process:
                if isinstance(step["text"],torch.Tensor):
                    step_list.append(step["text"])
            if len(step_list)>0:
                process_text = torch.stack(step_list)
                text.append(process_text)
            text.append(nutrition_analysis)
            recipe_image = self.recipe_id2tensor[recipe_id]["image"]
            process_pic = [step["picture"] for step in cur_process]
            image.append(recipe_image)
            image.extend(process_pic)

        ans["text"] = torch.cat(text).to(self.device)
        ans["image"] = torch.cat(image).to(self.device)
        return ans


print("loading dataset......")
dataset_path = model_dir = os.path.join(abs_dir, 'Dataset/dataset_file/dataset_10K')
dataset = torch.load(dataset_path)
dataset_20K_test_path = os.path.join(abs_dir, 'Dataset/dataset_file/dataset_test')
dataset_20K_test = torch.load(dataset_20K_test_path)
class IRDataset_Mul(Dataset):
    def __init__(self, flag=0):
        if flag == 0:
            self.train_dataset = dataset[0:len(dataset) * 8 // 10]  # read only
            self.dataset = self.train_dataset
        elif flag == 1:
            self.vaild_dataset = dataset[len(dataset) * 8 // 10: len(dataset) * 8 // 10 + 32]
            self.dataset = self.vaild_dataset
        elif flag == 2:
            self.test_dataset = dataset_20K_test
            self.dataset = self.test_dataset
        self.len = self.dataset.shape[0]
    def __getitem__(self, batch_index):
        batch_triple = self.dataset[batch_index]  # dataset的大结构必须是list或则tensor，小结构不需要多余的维度，实现大小结构对调
        return batch_triple
    def __len__(self):
        return self.len


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
def collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    # dict 拆进去，tensor加一个维度整合，基本数据类型变成tensor，list不做处理，保证最外面的list是batch，字符串也是
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):  # [tensor(1,128)*100] -> tensor(100,1,128)
        try:
            return torch.stack(batch, 0, out=None)  # 报错说明长度不一致？
        except:
            batch.sort(key=lambda x: len(x), reverse=True)
            seq_len = torch.tensor([s.size(0) for s in batch])  # 获取数据真实的长度
            return seq_len,pad_sequence(batch, batch_first=True, padding_value=0)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return collate_fn([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):  # 处理成tensor
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):  # [{}] 变成{[]}
        return {key: collate_fn([d[key] for d in batch]) for key in elem}
    elif isinstance(elem,list):  # 如果batch内的单个数据项还用list 说明里面没有对齐，那么batch是最外面的list
        return batch
    raise TypeError(default_collate_err_msg_format.format(elem_type))



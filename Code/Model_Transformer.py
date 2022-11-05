from Resource.model import *
from Resource.dataset_and_MKG import *
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import time
from tqdm import tqdm


class Multimodal_Transformer_Trainer:
    def __init__(self, root='/home/hmh', device="cuda:0", train_batch=16, valid_batch=16, UseCacha=False):

        # 文件夹/服务器位置,网络输出位置
        self.root = root
        self.device = device
        self.UseCacha = UseCacha
        self.margin = 0.1
        # 图谱 + 数据集
        print("加载训验测数据集...")
        self.Triple_dataloader_for_train = DataLoader(dataset=IRDataset_Mul(root=self.root, flag=0),
                                                      batch_size=train_batch,
                                                      shuffle=False, drop_last=True, collate_fn=collate_fn)
        self.Triple_dataloader_for_valid = DataLoader(dataset=IRDataset_Mul(root=self.root, flag=1),
                                                      batch_size=valid_batch,
                                                      shuffle=False, collate_fn=collate_fn)  # batch为1时没必要用collate_fn
        self.Triple_dataloader_for_test = DataLoader(dataset=IRDataset_Mul(root=self.root, flag=2),
                                                      batch_size=1,
                                                      shuffle=False, collate_fn=collate_fn)
        print("加载多模态知识图谱...")
        self.MKG = MKG_Mul(root=self.root)
        # 创建模型和优化器
        self.model = IRnet_Multimodal_Transformer(device = device)  # 加载  初始化零值
        if self.UseCacha:
            self.model.load_state_dict(torch.load("Output/Multimodal_Transformer/Mul_Transformer_j2k2_m8n8_top8.pkl"))  # j
        # self.optimizer = optim.SGD(  # 这一步必须确保模型参数在cpu
        #     self.model.parameters(), lr=1e-3,weight_decay=0.01)  # lr设置1e-2 : 检索
        self.optimizer = optim.SGD(  # 这一步必须确保模型参数在cpu
            self.model.parameters(), lr=1e-3)  # lr设置1e-2 : 检索
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=4, gamma=0.68)
        self.model.to(device)
        self.earlyS = EarlyStopping(patience=8)
    ###########################################################################################
    def Main(self):
        cudnn.benchmark = True
        min_MK = self.Valid()
        for epoch in range(50):
            self.Train()
            MK = self.Valid()
            self.scheduler.step()
            self.earlyS(MK)
            if self.earlyS.early_stop:
                print("早停了,已经五个epoch验证集损失没有下降了")
                break
            if MK < min_MK:  # 保留F1得分高的
                min_MK = MK
                filename = "Output/Multimodal_Transformer/Mul_Transformer_j2k2_m8n8_top8.pkl"
                torch.save(self.model.state_dict(), filename)
            print("epoch:",epoch,"LR:",self.optimizer.param_groups[0]["lr"])
        return
    def Train(self):
        # 在训练的时候
        self.model.train()
        for batch_data in tqdm(self.Triple_dataloader_for_train):

            ans_context = self.MKG.context_gen(batch_data)  # 把放进GPU的步骤拿出来
            ans,ans_mask,b_s,c_s = self.get_triple_attr(batch_data, ans_context)  # (bs,cs,attr max,dim)
            answer_label = self.ans_text2label(batch_data['answer_text'])
            answer_label = self.Data_to_cuda(batch_data,ans,ans_mask,answer_label)

            loss,coo,ans_loss,ans_acc = self.model(batch_data['question_tensor'], batch_data['image_tensor'], batch_data['query_mask'],
                                   ans, ans_mask, batch_data["triple_mask"],answer_label, b_s, c_s, True)


            self.optimizer.zero_grad()
            t_loss = loss + ans_loss
            # t_loss = ans_loss
            t_loss.backward()
            self.optimizer.step()
            # tqdm.write(" ans_loss:Acc: %.2f,Cls_loss: %.2f" % (ans_acc, ans_loss))
            F1 = self.cal_F1(coo, batch_data["triple_mask"])
            tqdm.write("F1: %.2f ,Tri_loss：%.2f ,Acc: %.2f,Cls_loss: %.2f" % (F1, loss, ans_acc, ans_loss))

        return 0
    def Valid(self):
        # 测试的时候要从主实体开始定位开始。采集
        self.model.eval()
        for batch_data in self.Triple_dataloader_for_valid:

            ans_context = self.MKG.context_gen(batch_data)  # 把放进GPU的步骤拿出来
            ans,ans_mask,b_s,c_s = self.get_triple_attr(batch_data, ans_context)  # (bs,cs,attr max,dim)
            answer_label = self.ans_text2label(batch_data['answer_text'])
            answer_label = self.Data_to_cuda(batch_data,ans,ans_mask,answer_label)

            loss,coo,ans_loss,ans_acc = self.model(batch_data['question_tensor'], batch_data['image_tensor'], batch_data['query_mask'],
                                   ans, ans_mask, batch_data["triple_mask"],answer_label, b_s, c_s, True)

            F1 = self.cal_F1(coo,batch_data["triple_mask"])
            print("V: F1: %.2f ,Tri_loss：%.2f ,Acc: %.2f,Cls_loss: %.2f" % (F1, loss, ans_acc, ans_loss))
            # print("V: ans_loss:Acc: %.2f,Cls_loss: %.2f" % (ans_acc, ans_loss))

        # return ans_loss  # 综合判断是否过拟合
        return ans_loss + loss
    def Test(self):
        self.model.eval()
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        hit = []
        i = 0
        for batch_data in tqdm(self.Triple_dataloader_for_test):
            # 第一步就是对于query做主实体检测
            # 然后判断主实体的正确性
            ans_context = self.MKG.context_gen_test(batch_data)  # 把放进GPU的步骤拿出来
            ans,ans_mask,b_s,c_s = self.get_triple_attr_test(batch_data, ans_context)  # (bs,cs,attr max,dim)
            answer_label = self.ans_text2label(batch_data['answer_text'])
            answer_label = self.Data_to_cuda(batch_data,ans,ans_mask,answer_label)

            loss,coo,ans_loss,ans_acc = self.model(batch_data['question_tensor'], batch_data['image_tensor'], batch_data['query_mask'],
                                   ans, ans_mask, batch_data["triple_mask"],answer_label, b_s, c_s, False)
            # 1代表不适合 代表Negative
            sorted_coo, sorted_index = torch.sort(coo, descending=True, dim=-1)
            sorted_index = sorted_index[0].tolist()
            i+=1
            if i ==681:
                print("kelejichi")
            if i ==789:
                print("kelejichi")
            print(batch_data["ans_name"][0][0])
            hit.append(sorted_index.index(0))

            if answer_label ==1 and ans_acc == 1:
                TN+=1
            elif answer_label ==1 and ans_acc == 0:
                FN+=1
            elif answer_label ==0 and ans_acc == 1:
                TP+=1
            elif answer_label ==0 and ans_acc == 0:
                FP+=1
        rec = TP/(TP+FN)
        pre = TP/(TP+FP)
        F1 = 2*rec*pre/(rec+pre)
        acc = (TP+TN)/(TP+TN+FP+FN)
        print("T: rec: %.3f, pre: %.3f, F1: %.3f, acc: %.3f" % (rec,pre,F1,acc))

        return

    ############################################################################################
    def get_triple_attr(self,batch_data,ans_context):
        ''' 从dataloader出来的数据【list嵌套形式】 处理成 GPU tensor
        # 想把数据处理成 tensor(100,100,max_attr_num,dim) + mask：tensor(100,100)  的形式
        # 但是部分数据无法处理成特征
        # 这一步衔接模型处理，所以输出三项

        : return
        # name: tensor(10000,max_len,dim)  mask:tensor(10000)
        # type: tensor(10000,max_len)  mask:tensor(10000)
        # context_text: tensor(10000,max_len,128) mask:tensor(10000)
        # context_pic: tensor(10000,max_len,512) mask:tensor(10000)
        '''

        # 整理出一个大的list
        b_s = len(batch_data["ans_name_tensor"])
        c_s = len(batch_data["ans_name_tensor"][0])
        ans_name = batch_data["ans_name_tensor"]
        ans_type = batch_data["ans_type"]
        ans_name_all = []
        ans_context_all = []
        ans_type_all = []
        for i in range(b_s):
            if i==0:
                ans_name_all = ans_name[0].copy()
                ans_context_all = ans_context[0].copy()
                ans_type_all = ans_type[0].copy()
            else:
                ans_name_all.extend(ans_name[i])  # 这一步的数量可能有问题？  在第二个epoch的时候报错？
                ans_context_all.extend(ans_context[i])
                ans_type_all.extend(ans_type[i])

        ans_type_all = self.type_process(ans_type_all)
        ans_context_text_all, ans_context_pic_all = self.context_process(ans_context_all)

        name_len = torch.tensor([s.size(0) for s in ans_name_all])
        ans_name_tensor = pad_sequence(ans_name_all, batch_first=True, padding_value=0)
        type_len = torch.tensor([s.size(0) for s in ans_type_all])
        ans_type_tensor = pad_sequence(ans_type_all, batch_first=True, padding_value=0)
        context_text_len = torch.tensor([s.size(0) for s in ans_context_text_all])
        ans_context_text_tensor = pad_sequence(ans_context_text_all, batch_first=True, padding_value=0)
        context_pic_len = torch.tensor([s.size(0) for s in ans_context_pic_all])
        ans_context_pic_tensor = pad_sequence(ans_context_pic_all, batch_first=True, padding_value=0)

        mask = torch.stack([name_len,type_len,context_text_len,context_pic_len])
        name_max_len = ans_name_tensor.shape[1]

        context_text_len = ans_context_text_tensor.shape[1]
        context_pic_len = ans_context_pic_tensor.shape[1]
        max_len = name_max_len + 1 + context_text_len + context_pic_len
        ans_mask_tem = torch.zeros([mask.shape[1], max_len])  # 这一步不要给模型做
        for i in range(ans_mask_tem.shape[0]):
            for j in range(mask[0][i]):
                ans_mask_tem[i][j] = 1
            ans_mask_tem[i][name_max_len] = 1
            for j in range(mask[2][i]):
                ans_mask_tem[i][j + name_max_len + 1] = 1
            for j in range(mask[3][i]):
                ans_mask_tem[i][j + name_max_len + 1 + context_text_len] = 1
        ans_mask_tem = ans_mask_tem.reshape([b_s, c_s, ans_mask_tem.shape[1]])
        return [ans_name_tensor, ans_type_tensor, ans_context_text_tensor, ans_context_pic_tensor], [ans_mask_tem,type_len],b_s,c_s
    def get_triple_attr_test(self,batch_data,ans_context):
        ''' 从dataloader出来的数据【list嵌套形式】 处理成 GPU tensor
        # 想把数据处理成 tensor(100,100,max_attr_num,dim) + mask：tensor(100,100)  的形式
        # 但是部分数据无法处理成特征
        # 这一步衔接模型处理，所以输出三项

        : return
        # name: tensor(10000,max_len,dim)  mask:tensor(10000)
        # type: tensor(10000,max_len)  mask:tensor(10000)
        # context_text: tensor(10000,max_len,128) mask:tensor(10000)
        # context_pic: tensor(10000,max_len,512) mask:tensor(10000)
        '''

        # 整理出一个大的list
        b_s = len(batch_data["test_ans_name_tensor"])
        c_s = len(batch_data["test_ans_name_tensor"][0])
        ans_name = batch_data["test_ans_name_tensor"]
        ans_type = batch_data["test_ans_type"]
        ans_name_all = []
        ans_context_all = []
        ans_type_all = []
        for i in range(b_s):
            if i==0:
                ans_name_all = ans_name[0].copy()
                ans_context_all = ans_context[0].copy()
                ans_type_all = ans_type[0].copy()
            else:
                ans_name_all.extend(ans_name[i])  # 这一步的数量可能有问题？  在第二个epoch的时候报错？
                ans_context_all.extend(ans_context[i])
                ans_type_all.extend(ans_type[i])

        ans_type_all = self.type_process(ans_type_all)
        ans_context_text_all, ans_context_pic_all = self.context_process(ans_context_all)

        name_len = torch.tensor([s.size(0) for s in ans_name_all])
        ans_name_tensor = pad_sequence(ans_name_all, batch_first=True, padding_value=0)
        type_len = torch.tensor([s.size(0) for s in ans_type_all])
        ans_type_tensor = pad_sequence(ans_type_all, batch_first=True, padding_value=0)
        context_text_len = torch.tensor([s.size(0) for s in ans_context_text_all])
        ans_context_text_tensor = pad_sequence(ans_context_text_all, batch_first=True, padding_value=0)
        context_pic_len = torch.tensor([s.size(0) for s in ans_context_pic_all])
        ans_context_pic_tensor = pad_sequence(ans_context_pic_all, batch_first=True, padding_value=0)

        mask = torch.stack([name_len,type_len,context_text_len,context_pic_len])
        name_max_len = ans_name_tensor.shape[1]
        type_max_len = ans_type_tensor.shape[1]
        context_text_len = ans_context_text_tensor.shape[1]
        context_pic_len = ans_context_pic_tensor.shape[1]
        max_len = name_max_len + 1 + context_text_len + context_pic_len
        ans_mask_tem = torch.zeros([mask.shape[1], max_len])  # 这一步不要给模型做
        for i in range(ans_mask_tem.shape[0]):
            for j in range(mask[0][i]):
                ans_mask_tem[i][j] = 1
            ans_mask_tem[i][name_max_len] = 1  # 这个1是留给模型中编码后的type信息
            for j in range(mask[2][i]):
                ans_mask_tem[i][j + name_max_len + 1] = 1
            for j in range(mask[3][i]):
                ans_mask_tem[i][j + name_max_len + 1 + context_text_len] = 1
        ans_mask_tem = ans_mask_tem.reshape([b_s, c_s, ans_mask_tem.shape[1]])
        return [ans_name_tensor, ans_type_tensor, ans_context_text_tensor, ans_context_pic_tensor], [ans_mask_tem,type_len],b_s,c_s
    def type_process(self,ans_type_all):
        candidate_s = len(ans_type_all)
        ans_type_tem = []
        for i in range(candidate_s):
            # 拆开来变成一个tensor
            type_list = ans_type_all[i][0]
            for index, triple in enumerate(ans_type_all[i]):  # [[1,2,3] [1,2,3]]
                if index != 0:
                    type_list.extend(triple)  # 将两个list合并
            ans_type_tem.append(torch.tensor(type_list))

        return ans_type_tem
    def context_process(self,ans_context_all):
        ans_context_text_all = []
        ans_context_pic_all = []
        for ans_context in ans_context_all:
            ans_context_text_all.append(ans_context["text"])
            ans_context_pic_all.append(ans_context["image"])
        return ans_context_text_all, ans_context_pic_all
    def Data_to_cuda(self, batch_data, ans, ans_mask,answer_label):
        # 比较合理的流程应该是：
        # 数据加载之后经过一个向量化的层
        # 规整之后统一进入GPU 同时做detach
        # 模型的输入应该是规整的GPU tensor
        for key in batch_data.keys():
            if isinstance(batch_data[key], torch.Tensor):
                batch_data[key] = batch_data[key].to(self.device).detach()
        for i in range(len(ans)):  # 还是包装成一整个tensor进去比较好
            ans[i] = ans[i].to(self.device).detach()
        for i in range(len(ans_mask)):
            ans_mask[i] = ans_mask[i].to(self.device).detach()
        answer_label = answer_label.to(self.device).detach().long()
        return answer_label
    def ans_text2label(self,b_text):
        bs = len(b_text)
        answer_label = torch.zeros([bs])
        for index in range(bs):
            if b_text[index]=="不适合":
                answer_label[index]=1
        return answer_label
    ############################################################################################
    def cal_F1(self,coo,label):
        # coo是预测的得分，label是标签
        # 获取我计算的得分的下标
        sorted_coo, sorted_index = torch.sort(coo, descending=True, dim=-1)
        pre_index = []
        for line in sorted_coo:
            for index,value in enumerate(line):
                if line[0]- value > self.margin:
                    pre_index.append(index)
                    break

        # 命中的/我预测的
        # 命中的/正确的总数
        pre_total = 0 + 0.1
        rec_total = 0 + 0.1
        hit = 0
        for i,last_index in enumerate(pre_index):
            pre_total+=last_index
            rec_total+=label[i]  # 标签是k，代表0 ,1 ,2 ,3 ....k-1是正确答案
            for j in range(last_index):
                if sorted_index[i][j]<label[i]:  # 比如label是1  那么只有0可以算命中
                    hit+=1

        precision = hit/pre_total
        recall = hit/rec_total

        if hit == 0:  # 防止除零
            F1 = 0
        else:
            F1 = 2 * precision * recall / (precision + recall)
        return F1

if __name__ == '__main__':

    tr = Multimodal_Transformer_Trainer(root='E:/code', device="cuda:0", train_batch=16, valid_batch=32, UseCacha=True)
    tr.Main()  #
    with torch.no_grad():
        tr.Test()

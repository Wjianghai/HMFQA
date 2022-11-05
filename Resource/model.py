from Util.Transformer_Module.transformer import TransformerBlock, SingleTransformerBlock, CrossTransformerBlock
from Util.modules.transformer import TransformerEncoder
from Util.conceptBert.fusion_modules.cti_model.fcnet import FCNet
from Util.conceptBert.fusion_modules.cti_model.tcnet import TCNet
from Util.conceptBert.fusion_modules.cti_model.triattention import TriAttention

import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from torch.nn import MultiLabelMarginLoss
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence ,pad_packed_sequence
import numpy as np
import warnings

INF = 1e20
warnings.simplefilter("ignore")

def create_mask(x, N, use_cuda=True):
    x = x.data
    mask = np.zeros((x.size(0), N))
    for i in range(x.size(0)):
        mask[i, :x[i]] = 1
    return to_cuda(torch.Tensor(mask), use_cuda)


########################################################################
class IRnet_Multimodal_Transformer(torch.nn.Module):
    def __init__(self,cfg):
        super(IRnet_Multimodal_Transformer, self).__init__()
        device = cfg["Trainer"]["device"]
        picture_dim = cfg["MODEL"]["picture_dim"]
        hidden_dim = cfg["MODEL"]["hidden_dim"]
        j = cfg["MODEL"]["J"]
        k = cfg["MODEL"]["K"]
        m = cfg["MODEL"]["M"]
        n = cfg["MODEL"]["N"]

        self.device = device
        self.picture_linear = nn.Linear(picture_dim, hidden_dim, bias=False)  # 512 * 128
        # multi-layers transformer blocks, deep network
        self.Self_transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden=hidden_dim, attn_heads=4, feed_forward_hidden=hidden_dim*2, dropout=0) for _ in range(n)])
        self.Single_transformer_blocks_for_ans = nn.ModuleList(
            [SingleTransformerBlock(hidden=hidden_dim, attn_heads=4, feed_forward_hidden=hidden_dim*2, dropout=0) for _ in range(m)])
        # 检索
        self.Single_transformer_blocks = nn.ModuleList(
            [SingleTransformerBlock(hidden=hidden_dim, attn_heads=4, feed_forward_hidden=hidden_dim*2, dropout=0) for _ in range(k)])
        self.Cross_transformer_blocks = nn.ModuleList(
            [CrossTransformerBlock(hidden=hidden_dim, attn_heads=4, feed_forward_hidden=hidden_dim*2, dropout=0) for _ in range(j)])
        self.lstm_enc_type = EncoderRNN(vocab_size = 7, embed_size = hidden_dim, hidden_size = hidden_dim,
                                        bidirectional=True,
                                        rnn_type='lstm',
                                        device = device)
        self.loss_fn = MultiLabelMarginLoss()
        self.prediction = SimpleClassifier(hidden_dim, hidden_dim*2, 2, 0.5)
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.top_k = cfg["MODEL"]["top_k"]


    def forward(self,batch_text,batch_image,que_mask,ans,ans_mask,label,ans_label,b_s,c_s,train_flag):
        # 输入三个部分。输入，输入掩码，标签。
        ###############################################################
        # preprocess: type embedding and image projection
        ans[1] = self.lstm_enc_type(ans[1], ans_mask[1])[1].unsqueeze(1)
        ans[3] = self.picture_linear(ans[3])
        ans = torch.cat(ans,dim=1)
        ans = ans.reshape([b_s,c_s,ans.shape[1],ans.shape[2]])
        query = torch.cat((batch_text.unsqueeze(1), self.picture_linear(batch_image).unsqueeze(1)), dim=1).squeeze(2)
        # wto Wstructure消解
        # ans = ans[:,:,2:,:]
        # ans_mask[0] = ans_mask[0][:,:,2:]
        # # wto Wtype消解
        # ans = torch.cat([ans[:,:,:2,:],ans[:,:,3:,:]],dim=2)
        # ans_mask[0] = torch.cat([ans_mask[0][:,:,:2],ans_mask[0][:,:,3:]],dim=2)
        # # wto Wcontext消解
        # ans = ans[:,:,:3,:]
        # ans_mask[0] = ans_mask[0][:,:,:3]
        ################################################################
        # 先进行检索，预测最高得分
        coo = []
        for i in range(b_s):
            # 构造n和不同问题和n个相同答案
            query_i = query[i,:,:].unsqueeze(0).expand([c_s,query.shape[1],query.shape[2]])
            query_mask_i = que_mask[i, :].unsqueeze(0).expand([c_s, que_mask.shape[1]])
            ans_i = ans[i]
            ans_mask_i = ans_mask[0][i]
            # 构造query和triple一起送进去 # [b,2, dim] [[b,att_num,dim] * 100]
            sco = self.Multimodal_Trans(query_i,ans_i,query_mask_i,ans_mask_i)
            coo.append(sco)
        coo = torch.stack(coo)  # [b,candidate_num] []
        loss = self.mul_label_loss(coo, label)
        ###############################################################
        # 训练时用GroundTruth做分类,测试时用结果
        if train_flag:  # 在训练和验证的时候保持一致
            gt_ans = ans[:,:self.top_k,:,:]  # [b_s,attr_num,128]
            gt_ans_mask = ans_mask[0][:,:self.top_k,:]  # # [b_s,attr_num]
        else:  # 测试的时候可以取少点
            highest_index = torch.sort(coo, descending=True)[1][:, :self.top_k]  # [bs,5]
            gt_ans = torch.stack([torch.index_select(ans[i],0,highest_index[i]) for i in range(b_s)])  # [k,attr,dim]
            gt_ans_mask = torch.stack([torch.index_select(ans_mask[0][i],0,highest_index[i]) for i in range(b_s)])  # [k,attr]
            # gt_ans = ans
            # gt_ans_mask = ans_mask[0]

        gt_ans = gt_ans.reshape([gt_ans.shape[0],-1,gt_ans.shape[3]])
        gt_ans_mask = gt_ans_mask.reshape([gt_ans_mask.shape[0],-1])

        ans_loss,ans_acc = self.Answer_model(query.detach(),gt_ans.detach(),gt_ans_mask,ans_label)
        return loss,coo,ans_loss,ans_acc  # 【batch,candidate_num】

    def Multimodal_Trans(self,Q_r,ans_i,que_mask,ans_mask_i):
        # 把ans_mask_i处理成01形式的GPU tensor

        # 提取线索
        clue = Q_r
        for transformer in self.Single_transformer_blocks:
            clue = transformer.forward(clue, ans_i, que_mask=que_mask, ans_mask=ans_mask_i)
            # [b,2, dim] [b,att_num,dim]  [b,2] [b,max_att_num]

        # 线索和query融合 打分  这里不需要mask吗？
        for transformer in self.Cross_transformer_blocks:
            clue,Q_r = transformer.forward(clue, Q_r, mask=que_mask)  # [b,2,dim] [b,2,dim]

        # 取第一项做LN-FC-ReLU-FC 效果奇差
        # sco = self.layer0(c_q_pair[:,0,:])
        sco = torch.sum(clue[:,0,:] * Q_r[:,0,:],dim=-1)

        return sco
    def Answer_model(self,query,ans,ans_mask,ans_label):
        # query（bs,2,128）  ans(bs,attr,128)  mask(bs,attr)
        # 用一个transformer整合一下
        #################################

        kg = query  # 是吧ans直接拼接进来提特征，。
        for transformer in self.Single_transformer_blocks_for_ans:
            kg = transformer.forward(kg, ans, que_mask=None, ans_mask=ans_mask)

        cls = torch.zeros([query.shape[0],1,query.shape[2]]).to(self.device)
        x = torch.cat((cls, query, kg), dim=1)
        que_mask = torch.ones([query.shape[0],1+2+2]).to(self.device)

        for transformer in self.Self_transformer_blocks:
            x = transformer.forward(x, que_mask)  # 具体来说是这一步   做聚合

        fea = x[:,0]
        pre = self.prediction(fea)  # [bs,2]
        # 一个正例得分一个反例得分
        return self.cls_loss(pre,ans_label)

    def mul_label_loss(self, coo, mask):
        ans_mask = torch.ones([coo.shape[0],coo.shape[1]]).to(self.device) * -1
        for i in range(coo.shape[0]):
            for j in range(mask[i]):
                ans_mask[i][j] = j
        loss = self.loss_fn(coo,ans_mask.long())  # 数据类型不对 x是float  后面是long

        return loss
    def cls_loss(self,pre,ans_label):
        bs = pre.shape[0]
        ans_loss = self.cls_loss_fn(pre,ans_label)
        hit = 0

        for i in range(bs):
            if ans_label[i]==0:  # 0代表适合 1代表不适合  说明0的得分要高些
                if pre[i][0] > pre[i][1]:
                    hit+=1
            else:
                if pre[i][1] > pre[i][0]:
                    hit+=1
        ans_acc = hit/bs
        return ans_loss,ans_acc
class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout=None, \
        bidirectional=False, rnn_type='lstm', device="cuda:0"):
        super(EncoderRNN, self).__init__()
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.device = device
        self.hidden_size = hidden_size // 2 if bidirectional else hidden_size
        self.num_directions = 2 if bidirectional else 1
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.embed.weight.data.uniform_(-0.08, 0.08)
        model = nn.LSTM
        self.model = model(embed_size, self.hidden_size, 1, batch_first=True, bidirectional=bidirectional)

    def forward(self, x, x_len):
        """x: [batch_size * max_length]
           x_len: [batch_size]
        """
        x = self.embed(x)
        if self.dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)

        sorted_x_len, indx = torch.sort(x_len, 0, descending=True)
        x = pack_padded_sequence(x[indx], sorted_x_len.data.tolist(), batch_first=True)

        h0 = self.to_cuda(torch.zeros(self.num_directions, x_len.size(0), self.hidden_size), self.device)
        c0 = self.to_cuda(torch.zeros(self.num_directions, x_len.size(0), self.hidden_size), self.device)

        packed_h, (packed_h_t, _) = self.model(x, (h0, c0))
        packed_h_t = torch.cat([packed_h_t[i] for i in range(packed_h_t.size(0))], -1)

        hh, _ = pad_packed_sequence(packed_h, batch_first=True)

        # restore the sorting
        _, inverse_indx = torch.sort(indx, 0)
        restore_hh = hh[inverse_indx]
        restore_packed_h_t = packed_h_t[inverse_indx]
        return restore_hh, restore_packed_h_t

    def to_cuda(self, x, device):
        x = x.to(device)
        return x
class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits
def to_cuda(x, device):
    x = x.to(device)
    return x


###############################################
class IRnet_BAM(torch.nn.Module):
    def __init__(self, num_hops=1, picture_dim = 512, hidden_dim=768,
                  hidden_size=768, att='add',device="cuda:0"):
        # hidden_dim表示最终输入和三元组的嵌入维度。
        super(IRnet_BAM, self).__init__()
        self.device = device
        self.batchnorm = nn.BatchNorm1d(hidden_size)

        # get query and ans
        self.picture_linear = nn.Linear(picture_dim, hidden_dim, bias=False)
        self.ans_key = nn.Linear(hidden_dim, hidden_size, bias=False)
        self.ans_val = nn.Linear(hidden_dim, hidden_size, bias=False)


        # 基于Q  提取KB
        self.num_hops = num_hops
        self.memory_hop = RomHop_batch(hidden_size, hidden_size, hidden_size, atten_type=att)
        self.lstm_enc_type = EncoderRNN(vocab_size = 7, embed_size = hidden_dim, hidden_size = hidden_dim,
                                        bidirectional=True,
                                        rnn_type='lstm',
                                        device = device)
        self.loss_fn = MultiLabelMarginLoss()
        # Answer 模块
        self.Self_transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden=hidden_dim, attn_heads=4, feed_forward_hidden=hidden_dim*2, dropout=0) for _ in range(8)])
        self.Single_transformer_blocks_for_ans = nn.ModuleList(
            [SingleTransformerBlock(hidden=hidden_dim, attn_heads=4, feed_forward_hidden=hidden_dim*2, dropout=0) for _ in range(8)])

        self.prediction = SimpleClassifier(hidden_dim, hidden_dim * 2, 2, 0.5)
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.top_k = 16

    def forward(self,batch_text,batch_image,que_mask,ans,ans_mask,label,ans_label,b_s,c_s,train_flag):
        # 1 KV化只用一个参数  2在q_att处做了修改（不用mem，sum去计算q_att/然后不用q_att去对齐问题/不用q_att去更新KV和问题）理由是没有word级别的特征

        # preprocess: type embedding and image projection and ans KV
        ans[1] = self.lstm_enc_type(ans[1], ans_mask[1])[1].unsqueeze(1)
        ans[3] = self.picture_linear(ans[3])
        ans = torch.cat(ans,dim=1)
        ans = ans.reshape([b_s,c_s,ans.shape[1],ans.shape[2]])
        query = torch.cat((batch_text.unsqueeze(1), self.picture_linear(batch_image).unsqueeze(1)), dim=1).squeeze(2)
        ans_key = self.ans_key(ans)
        ans_val = self.ans_val(ans)

        # importance + enhancing(去掉attr  更新问题表征) /（bs,word_num,dim） (bs,candidate_num,attr,dim)
        Q_r, ans_key, ans_val = self.memory_hop(query, ans_key, ans_val, ctx_mask = ans_mask[0], query_mask = que_mask)
        q_r = torch.sum(Q_r,dim=1)  # (这里模拟一个权重均等的q_attr)

        # Generalization module  循环推理
        for _ in range(self.num_hops):
            q_r_tmp = self.memory_hop.gru_step(q_r, ans_key, ans_val)  # [batch,128]  [batch,candidate_num,128]
            q_r = self.batchnorm(q_r + q_r_tmp)

        # # 各个问题的周围100候选和    各个问题候选对交叉损失
        coo = torch.bmm(ans_key, q_r.unsqueeze(2)).squeeze(2)
        loss = self.mul_label_loss(coo, label)  # 正例和反例的对比损失
        ###############################################################
        # 训练时用GroundTruth做分类,测试时用结果
        if train_flag:  # 在训练和验证的时候保持一致
            gt_ans = ans[:,:self.top_k,:,:]  # [b_s,attr_num,128]
            gt_ans_mask = ans_mask[0][:,:self.top_k,:]  # # [b_s,attr_num]
        else:  # 测试的时候标志为 Fasle,取前k项做
            highest_index = torch.sort(coo, descending=True)[1][:, :self.top_k]  # [bs,5]
            gt_ans = torch.stack([torch.index_select(ans[i],0,highest_index[i]) for i in range(b_s)])  # [k,attr,dim]
            gt_ans_mask = torch.stack([torch.index_select(ans_mask[0][i],0,highest_index[i]) for i in range(b_s)])  # [k,attr]
            # 这不比gather好用多了
        gt_ans = gt_ans.reshape([gt_ans.shape[0],-1,gt_ans.shape[3]])
        gt_ans_mask = gt_ans_mask.reshape([gt_ans_mask.shape[0],-1])

        ans_loss,ans_acc = self.Answer_model(query.detach(),gt_ans.detach(),gt_ans_mask,ans_label)

        return loss,coo,ans_loss,ans_acc  # 【batch,candidate_num】

    def Answer_model(self, query, ans, ans_mask, ans_label):
        # query（bs,2,128）  ans(bs,attr,128)  mask(bs,attr)
        # 用一个transformer整合一下
        #################################

        kg = query  # 是吧ans直接拼接进来提特征，。
        for transformer in self.Single_transformer_blocks_for_ans:
            kg = transformer.forward(kg, ans, que_mask=None, ans_mask=ans_mask)

        cls = torch.zeros([query.shape[0], 1, query.shape[2]]).to(self.device)
        x = torch.cat((cls, query, kg), dim=1)
        que_mask = torch.ones([query.shape[0], 1 + 2 + 2]).to(self.device)

        for transformer in self.Self_transformer_blocks:
            x = transformer.forward(x, que_mask)  # 具体来说是这一步   做聚合

        fea = x[:, 0]
        pre = self.prediction(fea)  # [bs,2]
        # 一个正例得分一个反例得分
        return self.cls_loss(pre, ans_label)
    def cls_loss(self,pre,ans_label):
        bs = pre.shape[0]
        ans_loss = self.cls_loss_fn(pre,ans_label)
        hit = 0

        for i in range(bs):
            if ans_label[i]==0:  # 0代表适合 1代表不适合  说明0的得分要高些
                if pre[i][0] > pre[i][1]:
                    hit+=1
            else:
                if pre[i][1] > pre[i][0]:
                    hit+=1
        ans_acc = hit/bs
        return ans_loss,ans_acc
    def mul_label_loss(self, coo, mask):
        ans_mask = torch.ones([coo.shape[0],coo.shape[1]]).to(self.device) * -1
        for i in range(coo.shape[0]):
            for j in range(mask[i]):
                ans_mask[i][j] = j
        loss = self.loss_fn(coo,ans_mask.long())  # 数据类型不对 x是float  后面是long

        return loss
class RomHop_batch(nn.Module):
    def __init__(self, query_embed_size, in_memory_embed_size, hidden_size, atten_type='add'):
        super(RomHop_batch, self).__init__()
        self.hidden_size = hidden_size
        self.gru_linear_z = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.gru_linear_r = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.gru_linear_t = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.gru_atten = Attention(hidden_size, query_embed_size, in_memory_embed_size, atten_type=atten_type)

    def forward(self, query_embed, in_memory_embed, out_memory_embed,ctx_mask=None, query_mask=None):
        output = self.update_coatt_cat_maxpool(query_embed, in_memory_embed, out_memory_embed,\
                                               ctx_mask=ctx_mask, query_mask=query_mask)
        return output

    def gru_step(self, h_state, in_memory_embed, out_memory_embed, atten_mask=None):
        attention = self.gru_atten(h_state, in_memory_embed, atten_mask=atten_mask)
        probs = torch.softmax(attention, dim=-1)

        memory_output = torch.bmm(probs.unsqueeze(1), out_memory_embed).squeeze(1)
        # GRU-like memory update
        z = torch.sigmoid(self.gru_linear_z(torch.cat([h_state, memory_output], -1)))
        r = torch.sigmoid(self.gru_linear_r(torch.cat([h_state, memory_output], -1)))
        t = torch.tanh(self.gru_linear_t(torch.cat([r * h_state, memory_output], -1)))
        output = (1 - z) * h_state + z * t
        return output

    def update_coatt_cat_maxpool(self, query_embed, in_memory_embed, out_memory_embed, ctx_mask=None, query_mask=None):

        # (batch_num, word_num, candidate_num, attr_num)
        attention = torch.bmm(query_embed, in_memory_embed.reshape(in_memory_embed.size(0), -1, in_memory_embed.size(-1)) \
                              .transpose(1, 2)).reshape(query_embed.size(0), query_embed.size(1), in_memory_embed.size(1),-1)
        if ctx_mask is not None:
            attention = ctx_mask.unsqueeze(1) * attention - (1 - ctx_mask).unsqueeze(1) * INF
        if query_mask is not None:
            attention = query_mask.unsqueeze(2).unsqueeze(-1) * attention - (1 - query_mask).unsqueeze(2).unsqueeze(-1) * INF


        # Importance module
        # (batch_num ,candidate_num, attr_num)
        # 把ans的attr属性消除掉  这里可以手动调节权重配比
        kb_feature_att = F.max_pool1d(attention.reshape(attention.size(0), attention.size(1), -1).transpose(1, 2),
                                      kernel_size=attention.size(1)).squeeze(-1).reshape(attention.size(0), -1,
                                                                                      attention.size(-1))
        ###################################################################################################################
        # 手动固定参数
        mask = torch.ones([kb_feature_att.shape[0],kb_feature_att.shape[1],kb_feature_att.shape[2]]).to(query_embed.device)
        if ctx_mask is not None:
            mask = ctx_mask * mask - (1 - ctx_mask) * INF
        kb_feature_att = torch.softmax(mask, dim=-1).reshape(-1, kb_feature_att.size(-1)).unsqueeze(1)

        # kb_feature_att = torch.softmax(kb_feature_att, dim=-1).reshape(-1, kb_feature_att.size(-1)).unsqueeze(1)
        in_memory_embed = torch.bmm(kb_feature_att, in_memory_embed.reshape(-1, in_memory_embed.size(2),in_memory_embed.size(-1))).\
            squeeze(1).reshape(in_memory_embed.size(0), in_memory_embed.size(1), -1)
        out_memory_embed = out_memory_embed.sum(2)


        # Enhanced module
        # 去掉属性之后ans 用V来更新query  用query来更新K
        attention = F.max_pool1d(attention.view(attention.size(0), -1, attention.size(-1)),kernel_size=attention.size(-1))\
            .squeeze(-1).view(attention.size(0), attention.size(1),attention.size(2))
        # 先合并无关维度，进行池化处理之后，再做维度展开
        probs = torch.softmax(attention, dim=-1)
        new_query_embed = query_embed + torch.bmm(probs, out_memory_embed)

        probs2 = torch.softmax(attention, dim=1)
        in_memory_embed = in_memory_embed + torch.bmm(probs2.transpose(1, 2), new_query_embed)
        return new_query_embed, in_memory_embed, out_memory_embed
class Attention(nn.Module):
    def __init__(self, hidden_size, h_state_embed_size=None, in_memory_embed_size=None, atten_type='simple'):
        super(Attention, self).__init__()
        self.atten_type = atten_type
        if not h_state_embed_size:
            h_state_embed_size = hidden_size
        if not in_memory_embed_size:
            in_memory_embed_size = hidden_size
        if atten_type in ('mul', 'add'):
            self.W = torch.Tensor(h_state_embed_size, hidden_size)
            self.W = nn.Parameter(nn.init.xavier_uniform_(self.W))
            if atten_type == 'add':
                self.W2 = torch.Tensor(in_memory_embed_size, hidden_size)
                self.W2 = nn.Parameter(nn.init.xavier_uniform_(self.W2))
                self.W3 = torch.Tensor(hidden_size, 1)
                self.W3 = nn.Parameter(nn.init.xavier_uniform_(self.W3))
        elif atten_type == 'simple':
            pass
        else:
            raise RuntimeError('Unknown atten_type: {}'.format(self.atten_type))

    def forward(self, query_embed, in_memory_embed, atten_mask=None):
        if self.atten_type == 'simple':  # simple attention
            attention = torch.bmm(in_memory_embed, query_embed.unsqueeze(2)).squeeze(2)
        elif self.atten_type == 'mul':  # multiplicative attention
            attention = torch.bmm(in_memory_embed, torch.mm(query_embed, self.W).unsqueeze(2)).squeeze(2)
        elif self.atten_type == 'add':  # additive attention
            attention = torch.tanh(
                (in_memory_embed @ self.W2)  # 【batch,ans_token,dim】
                + torch.mm(query_embed, self.W).unsqueeze(1))  # # 【batch,1,dim】
            attention = torch.mm(attention.view(-1, attention.size(-1)), self.W3).view(attention.size(0), -1)  # 【batch,ans_token】
        else:
            raise RuntimeError('Unknown atten_type: {}'.format(self.atten_type))

        if atten_mask is not None:
            # Exclude masked elements from the
            attention = atten_mask * attention - (1 - atten_mask) * INF
        return attention
###############################################
class IRnet_ConceptBert(torch.nn.Module):
    def __init__(self, picture_dim=512, hidden_dim=768, device = "cuda:0"):
        self.device = device
        super(IRnet_ConceptBert, self).__init__()
        self.picture_linear = nn.Linear(picture_dim, hidden_dim, bias=False)
        self.Single_transformer_blocks = nn.ModuleList(
            [SingleTransformerBlock(hidden=hidden_dim, attn_heads=4, feed_forward_hidden=hidden_dim*2, dropout=0) for _ in range(1)])
        self.lstm_enc_type = EncoderRNN(vocab_size = 7, embed_size = hidden_dim, hidden_size = hidden_dim,
                                        bidirectional=True,
                                        rnn_type='lstm',
                                        device = device)

        self.loss_fn = MultiLabelMarginLoss()
        self.prediction = SimpleClassifier_Concept(128, 128 * 2, 2, 0.5)

        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.aggregator = CTIModel(
            v_dim=512,
            q_dim=768,
            kg_dim=768,
            glimpse=2,
            h_dim=64,
            h_out=1,
            rank=32,
            k=1,
        )

    def forward(self,batch_text,batch_image,que_mask,ans,ans_mask,label,ans_label,b_s,c_s,train_flag):
        # 输入三个部分。输入，输入掩码，标签。
        ###############################################################
        # preprocess: type embedding and image projection
        ans[1] = self.lstm_enc_type(ans[1], ans_mask[1])[1].unsqueeze(1)
        ans[3] = self.picture_linear(ans[3])
        ans = torch.cat(ans,dim=1)
        ans = ans.reshape([ans.shape[0], -1, ans.shape[2]])  # 整合中间两个维度
        ans_mask = ans_mask[0].reshape([b_s,-1])
        # ans = ans.reshape([b_s,c_s, ans.shape[1], ans.shape[2]])
        # ans = ans[:,0,:,:]
        # ans_mask = ans_mask[0][:, 0, :]
        #######################################################################
        # Send the image, question and ConceptNet to the Aggregator module
        kg = batch_text
        for transformer in self.Single_transformer_blocks:
            kg = transformer.forward(kg, ans, que_mask=None, ans_mask=ans_mask)
        # [b,2, dim] [b,att_num,dim]  [b,2] [b,max_att_num]

        result_vector, result_attention = self.aggregator(
            batch_image, batch_text, kg,
        )
        # v: [batch, num_objs, obj_dim]
        # b: [batch, num_objs, b_dim]
        # q: [batch_size, seq_length]
        pre = self.prediction(result_vector)
        # Send the vector to the SimpleClassifier to get the answer
        return self.cls_loss(pre, ans_label)

    def cls_loss(self,pre,ans_label):
        bs = pre.shape[0]
        ans_loss = self.cls_loss_fn(pre,ans_label)
        hit = 0

        for i in range(bs):
            if ans_label[i]==0:  # 0代表适合 1代表不适合  说明0的得分要高些
                if pre[i][0] > pre[i][1]:
                    hit+=1
            else:
                if pre[i][1] > pre[i][0]:
                    hit+=1
        ans_acc = hit/bs
        return ans_loss,ans_acc
class CTIModel(nn.Module):
    """
        Instance of a Compact Trilinear Interaction model (see https://arxiv.org/pdf/1909.11874.pdf)
    """

    def __init__(
        self, v_dim, q_dim, kg_dim, glimpse, h_dim=512, h_out=1, rank=32, k=1,
    ):
        super(CTIModel, self).__init__()

        self.glimpse = glimpse

        self.t_att = TriAttention(
            v_dim, q_dim, kg_dim, h_dim, 1, rank, glimpse, k, dropout=[0.2, 0.5, 0.2],
        )

        t_net = []
        q_prj = []
        kg_prj = []
        for _ in range(glimpse):
            t_net.append(
                TCNet(
                    v_dim,
                    q_dim,
                    kg_dim,
                    h_dim,
                    h_out,
                    rank,
                    1,
                    k=2,
                    dropout=[0.2, 0.5, 0.2],
                )
            )
            q_prj.append(FCNet([h_dim * 2, h_dim * 2], "", 0.2))
            kg_prj.append(FCNet([h_dim * 2, h_dim * 2], "", 0.2))

        self.t_net = nn.ModuleList(t_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.kg_prj = nn.ModuleList(kg_prj)

        self.q_pooler = FCNet([q_dim, h_dim * 2])
        self.kg_pooler = FCNet([kg_dim, h_dim * 2])

    # def forward(self, v, q, kg):
    def forward(self, v_emb, q_emb_raw, kg_emb_raw):
        """
            v: [batch, num_objs, obj_dim]
            b: [batch, num_objs, b_dim]
            q: [batch_size, seq_length]
        """
        b_emb = [0] * self.glimpse
        att, logits = self.t_att(v_emb, q_emb_raw, kg_emb_raw)

        q_emb = self.q_pooler(q_emb_raw)
        kg_emb = self.kg_pooler(kg_emb_raw)

        for g in range(self.glimpse):
            b_emb[g] = self.t_net[g].forward_with_weights(
                v_emb, q_emb_raw, kg_emb_raw, att[:, :, :, :, g]
            )

            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb
            kg_emb = self.kg_prj[g](b_emb[g].unsqueeze(1)) + kg_emb

        joint_emb = q_emb.sum(1) + kg_emb.sum(1)
        return joint_emb, att
class SimpleClassifier_Concept(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier_Concept, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None),
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits
###############################################
class HypergraphTransformer(nn.Module):
    def __init__(self, cfg):
        super(HypergraphTransformer, self).__init__()

        self.cfg = cfg

        self.n_hidden = cfg["MODEL"]["NUM_HIDDEN"]
        self.n_out = cfg["MODEL"]["NUM_OUT"]


        self.trans_k_with_q = self.get_network(self_type="kq")
        self.trans_q_with_k = self.get_network(self_type="qk")

        self.trans_k_mem = self.get_network(self_type="k_mem", layers=3)
        self.trans_q_mem = self.get_network(self_type="q_mem", layers=3)

        self.out_dropout = 0.0

        self.proj1 = nn.Linear(2 * self.n_hidden, self.n_hidden)
        self.proj2 = nn.Linear(self.n_hidden, self.n_out)

        self.picture_linear = nn.Linear(512, 768, bias=False)
        self.Single_transformer_blocks = nn.ModuleList(
            [SingleTransformerBlock(hidden=768, attn_heads=4, feed_forward_hidden=768*2, dropout=0) for _ in range(2)])
        self.lstm_enc_type = EncoderRNN(vocab_size = 7, embed_size = 768, hidden_size = 768,
                                        bidirectional=True,
                                        rnn_type='lstm',
                                        device = cfg["Trainer"]["device"])

    def get_network(self, self_type="", layers=-1):
        if self_type in ["kq", "k_mem"]:
            embed_dim, attn_dropout = self.n_hidden, self.cfg["MODEL"]["ATTN_DROPOUT_K"]
        elif self_type in ["qk", "q_mem"]:
            embed_dim, attn_dropout = self.n_hidden, self.cfg["MODEL"]["ATTN_DROPOUT_Q"]
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=self.cfg["MODEL"]["NUM_HEAD"],
            layers=max(self.cfg["MODEL"]["NUM_LAYER"], layers),
            attn_dropout=attn_dropout,
            relu_dropout=self.cfg["MODEL"]["RELU_DROPOUT"],
            res_dropout=self.cfg["MODEL"]["RES_DROPOUT"],
            embed_dropout=self.cfg["MODEL"]["EMB_DROPOUT"],
            attn_mask=self.cfg["MODEL"]["ATTN_MASK"],
            fc_hid_coeff=self.cfg["MODEL"]["FC_HID_COEFF"],
        )

    def forward(self,batch_text,batch_image,que_mask,ans,ans_mask,label,ans_label,b_s,c_s,train_flag):

        ans[1] = self.lstm_enc_type(ans[1], ans_mask[1])[1].unsqueeze(1)
        ans[3] = self.picture_linear(ans[3])
        ans = torch.cat(ans,dim=1)
        ans = ans.reshape([ans.shape[0], -1, ans.shape[2]])  # 整合中间两个维度
        ans_mask = ans_mask[0].reshape([b_s,-1])
        kg = batch_text
        for transformer in self.Single_transformer_blocks:
            kg = transformer.forward(kg, ans, que_mask=None, ans_mask=ans_mask)

        he_ques = batch_text
        he_kg = kg

        he_ques = he_ques.permute(1, 0, 2)
        he_kg = he_kg.permute(1, 0, 2)

        h_k_with_q = self.trans_k_with_q(he_kg, he_ques, he_ques)
        h_ks = self.trans_k_mem(h_k_with_q)
        h_ks_sum = torch.sum(h_ks, axis=0)

        h_q_with_k = self.trans_q_with_k(he_ques, he_kg, he_kg)
        h_qs = self.trans_q_mem(h_q_with_k)
        h_qs_sum = torch.sum(h_qs, axis=0)

        last_kq = torch.cat([h_ks_sum, h_qs_sum], dim=1)

        output = self.proj2(
            F.dropout(
                F.relu(self.proj1(last_kq)),
                p=self.out_dropout,
                training=self.training,
            )
        )
        pred = F.log_softmax(output, dim=1)
        pred_score, pred_ans = pred.max(1)
        ans_loss = F.nll_loss(pred, ans_label)  # 去负号求和取平均，所以对应位置数字越大越好
        hit = 0
        bs = output.shape[0]
        for i in range(bs):
            if pred_ans[i]==ans_label[i]:
                hit+=1
        ans_acc = hit / bs
        return ans_loss,ans_acc
class HAN(nn.Module):
    def __init__(self, cfg):
        super(HAN, self).__init__()

        self.cfg = cfg

        self.n_hidden = cfg["MODEL"]["NUM_HIDDEN"]
        self.n_head = cfg["MODEL"]["NUM_HEAD"]
        self.word_emb_size = cfg["MODEL"]["NUM_WORD_EMB"]
        self.n_out = cfg["MODEL"]["NUM_OUT"]

        self.h2att = torch.nn.Linear(self.n_hidden, self.n_head)
        self.softmax_att = torch.nn.Softmax(dim=2)
        self.fc_out = torch.nn.Linear(self.n_hidden * self.n_head, self.n_out)

        self.picture_linear = nn.Linear(512, 768, bias=False)
        self.Single_transformer_blocks = nn.ModuleList(
            [SingleTransformerBlock(hidden=768, attn_heads=4, feed_forward_hidden=768*2, dropout=0) for _ in range(2)])
        self.lstm_enc_type = EncoderRNN(vocab_size = 7, embed_size = 768, hidden_size = 768,
                                        bidirectional=True,
                                        rnn_type='lstm',
                                        device = cfg["Trainer"]["device"])

    def forward(self,batch_text,batch_image,que_mask,ans,ans_mask,label,ans_label,b_s,c_s,train_flag):

        ans[1] = self.lstm_enc_type(ans[1], ans_mask[1])[1].unsqueeze(1)
        ans[3] = self.picture_linear(ans[3])
        ans = torch.cat(ans,dim=1)
        ans = ans.reshape([ans.shape[0], -1, ans.shape[2]])  # 整合中间两个维度
        ans_mask = ans_mask[0].reshape([b_s,-1])
        kg = batch_text
        for transformer in self.Single_transformer_blocks:
            kg = transformer.forward(kg, ans, que_mask=None, ans_mask=ans_mask)

        he_ques = batch_text
        he_src = kg

        num_batch = he_ques.shape[0]
        num_he_ques = he_ques.shape[1]
        num_he_src = he_src.shape[1]

        he_ques = he_ques.permute(0, 2, 1)
        he_src = he_src.permute(0, 2, 1)

        he_ques_selfatt = he_ques.unsqueeze(3)
        he_src_selfatt = he_src.unsqueeze(2)

        self_mul = torch.matmul(he_ques_selfatt, he_src_selfatt)
        self_mul = self_mul.permute(0, 2, 3, 1)

        att_map = self.h2att(self_mul)
        att_map = att_map.permute(0, 3, 1, 2)

        att_map = torch.reshape(att_map, (-1, self.n_head, num_he_ques * num_he_src))
        att_map = self.softmax_att(att_map)
        att_map = torch.reshape(att_map, (-1, self.n_head, num_he_ques, num_he_src))
        he_ques = he_ques.unsqueeze(2)
        he_src = he_src.unsqueeze(3)

        for i in range(self.n_head):
            att_g = att_map[:, i : i + 1, :, :]
            att_g_t = att_g.repeat([1, self.n_hidden, 1, 1])
            att_out = torch.matmul(he_ques, att_g_t)
            att_out = torch.matmul(att_out, he_src)
            att_out = att_out.squeeze(-1)
            att_out_sq = att_out.squeeze(-1)

            if i == 0:
                output = att_out_sq
            else:
                output = torch.cat((output, att_out_sq), dim=1)

        output = self.fc_out(output)

        pred = F.log_softmax(output, dim=1)
        pred_score, pred_ans = pred.max(1)
        ans_loss = F.nll_loss(pred, ans_label)  # 去负号求和取平均，所以对应位置数字越大越好
        hit = 0
        bs = output.shape[0]
        for i in range(bs):
            if pred_ans[i] == ans_label[i]:
                hit += 1
        ans_acc = hit / bs
        return ans_loss, ans_acc
class BAN(nn.Module):
    def __init__(self, cfg):
        super(BAN, self).__init__()

        self.cfg = cfg

        self.n_hidden = cfg["MODEL"]["NUM_HIDDEN"]
        self.n_head = cfg["MODEL"]["NUM_HEAD"]
        self.n_out = cfg["MODEL"]["NUM_OUT"]


        self.h2att = torch.nn.Linear(self.n_hidden, self.n_head)
        self.softmax_att = torch.nn.Softmax(dim=2)
        self.fc_out = torch.nn.Linear(self.n_hidden * self.n_head, self.n_out)

        self.picture_linear = nn.Linear(512, 768, bias=False)
        self.Single_transformer_blocks = nn.ModuleList(
            [SingleTransformerBlock(hidden=768, attn_heads=4, feed_forward_hidden=768*2, dropout=0) for _ in range(2)])
        self.lstm_enc_type = EncoderRNN(vocab_size = 7, embed_size = 768, hidden_size = 768,
                                        bidirectional=True,
                                        rnn_type='lstm',
                                        device = cfg["Trainer"]["device"])

    def forward(self,batch_text,batch_image,que_mask,ans,ans_mask,label,ans_label,b_s,c_s,train_flag):
        ans[1] = self.lstm_enc_type(ans[1], ans_mask[1])[1].unsqueeze(1)
        ans[3] = self.picture_linear(ans[3])
        ans = torch.cat(ans, dim=1)
        ans = ans.reshape([ans.shape[0], -1, ans.shape[2]])  # 整合中间两个维度
        ans_mask = ans_mask[0].reshape([b_s, -1])
        kg = batch_text
        for transformer in self.Single_transformer_blocks:
            kg = transformer.forward(kg, ans, que_mask=None, ans_mask=ans_mask)

        he_ques = batch_text
        he_src = kg

        num_he_ques = he_ques.shape[1]
        num_he_src = he_src.shape[1]

        he_ques = he_ques.permute(0, 2, 1)
        he_src = he_src.permute(0, 2, 1)

        he_ques_selfatt = he_ques.unsqueeze(3)
        he_src_selfatt = he_src.unsqueeze(2)

        self_mul = torch.matmul(he_ques_selfatt, he_src_selfatt)
        self_mul = self_mul.permute(0, 2, 3, 1)

        att_map = self.h2att(self_mul)
        att_map = att_map.permute(0, 3, 1, 2)

        att_map = torch.reshape(att_map, (-1, self.n_head, num_he_ques * num_he_src))
        att_map = self.softmax_att(att_map)
        att_map = torch.reshape(att_map, (-1, self.n_head, num_he_ques, num_he_src))
        he_ques = he_ques.unsqueeze(2)
        he_src = he_src.unsqueeze(3)

        for i in range(self.n_head):
            att_g = att_map[:, i : i + 1, :, :]
            att_g_t = att_g.repeat([1, self.n_hidden, 1, 1])
            att_out = torch.matmul(he_ques, att_g_t)
            att_out = torch.matmul(att_out, he_src)
            att_out = att_out.squeeze(-1)
            att_out_sq = att_out.squeeze(-1)

            if i == 0:
                output = att_out_sq
            else:
                output = torch.cat((output, att_out_sq), dim=1)

        output = self.fc_out(output)

        pred = F.log_softmax(output, dim=1)
        pred_score, pred_ans = pred.max(1)
        ans_loss = F.nll_loss(pred, ans_label)  # 去负号求和取平均，所以对应位置数字越大越好
        hit = 0
        bs = output.shape[0]
        for i in range(bs):
            if pred_ans[i] == ans_label[i]:
                hit += 1
        ans_acc = hit / bs
        return ans_loss, ans_acc

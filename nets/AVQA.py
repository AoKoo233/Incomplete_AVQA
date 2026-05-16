import torch
# import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import copy
import math


from nets.bert_module import BertLMPredictionHead, BertLayer_Cross
from easydict import EasyDict as EDict

from nets.model import FMOT, AGE, AugEncoder

def cosine_similarity(a, b):

    a_norm = torch.linalg.norm(a, dim = -1, keepdim = True)
    a = a / a_norm

    b_norm = torch.linalg.norm(b, dim = -1, keepdim = True)
    b = b / b_norm

    return  torch.mm(a, b.t())

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TemporalSampling(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.class_embedding = nn.Parameter((width ** -0.5) * torch.randn(width))
        self.positional_embedding = nn.Parameter((width ** -0.5) * torch.randn(100, width))
        self.bert_config = EDict(
            num_attention_heads=8,
            hidden_size=width,
            attention_head_size=width,
            attention_probs_dropout_prob=0.1,
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.1,
            intermediate_size=width,
            vocab_size=42,
            num_layers=2
        )
        self.layer_ca = nn.ModuleList([BertLayer_Cross(self.bert_config) for _ in range(self.bert_config.num_layers)])
        self.head = BertLMPredictionHead(self.bert_config)

    def forward(self, x, query=None):
        #x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze().unsqueeze(0)
        for i in range(self.bert_config.num_layers):
            x, _ = self.layer_ca[i](x, query)
        
        logits = self.head(x).squeeze()
        return logits

class SpatialActivation(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.vocab_size = 42
        self.class_embedding = nn.Parameter((width ** -0.5) * torch.randn(width))
        self.positional_embedding = nn.Parameter((width ** -0.5) * torch.randn(100, width))
        self.bert_config = EDict(
            num_attention_heads=8,
            hidden_size=width,
            attention_head_size=width,
            attention_probs_dropout_prob=0.1,
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.1,
            intermediate_size=width,
            vocab_size=42,
            num_layers=2
        )
        self.layer_ca = nn.ModuleList([BertLayer_Cross(self.bert_config) for _ in range(self.bert_config.num_layers)])
        self.head = BertLMPredictionHead(self.bert_config)

    def forward(self, input, init_q=None):
        #input = input.permute(0, 2, 3, 1)
        #x = input.reshape(input.size(0), -1, 256)
        #query = torch.zeros(x.size(0), 1, x.size(-1)).to(x.device) if init_q is None else init_q.repeat(x.size(0), 1, 1)
        x = input
        query = init_q
        for i in range(self.bert_config.num_layers):
            query, att_map = self.layer_ca[i](query, x)
        att_map = att_map.sum(1).squeeze(1).sigmoid()
        att_map = (att_map - att_map.min(dim=1, keepdim=True)[0]) / (att_map.max(dim=1, keepdim=True)[0] - att_map.min(dim=1, keepdim=True)[0])
        
        #logits = self.head(query).mean(0)
        logits = self.head(query).squeeze()
        #return logits, att_map
        return logits


class QstLstmEncoder(nn.Module):

    def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size):

        super(QstLstmEncoder, self).__init__()
        self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(2*num_layers*hidden_size, embed_size)     # 2 for hidden and cell states

    def forward(self, question):

        qst_vec = self.word2vec(question)                             # [batch_size, max_qst_length=30, word_embed_size=300]
        qst_vec = self.tanh(qst_vec)
        qst_vec = qst_vec.transpose(0, 1)                             # [max_qst_length=30, batch_size, word_embed_size=300]
        self.lstm.flatten_parameters()
        _, (hidden, cell) = self.lstm(qst_vec)                        # [num_layers=2, batch_size, hidden_size=512]
        qst_feature = torch.cat((hidden, cell), 2)                    # [num_layers=2, batch_size, 2*hidden_size=1024]
        qst_feature = qst_feature.transpose(0, 1)                     # [batch_size, num_layers=2, 2*hidden_size=1024]
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
        qst_feature = self.tanh(qst_feature)
        qst_feature = self.fc(qst_feature)                            # [batch_size, embed_size]

        return qst_feature


class AVClipAttn(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(AVClipAttn, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cm_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src_q, src_v, src_mask=None, src_key_padding_mask=None):

        src_q = src_q.permute(1, 0, 2)
        src_v = src_v.permute(1, 0, 2)
        src1 = self.cm_attn(src_q, src_v, src_v, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]
        src2 = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]

        src_q = src_q + self.dropout11(src1) + self.dropout12(src2)
        src_q = self.norm1(src_q)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src_q))))
        src_q = src_q + self.dropout2(src2)
        src_q = self.norm2(src_q)

        return src_q.permute(1, 0, 2)






class AVHanLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(AVHanLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cm_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src_q, src_v, src_mask=None, src_key_padding_mask=None):

        src_q = src_q.permute(1, 0, 2)
        src_v = src_v.permute(1, 0, 2)
        src1 = self.cm_attn(src_q, src_v, src_v, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src2 = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        weight_cross = self.cm_attn(src_q, src_v, src_v, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[1]
        weight_self = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[1]
        src_q = src_q + self.dropout11(src1) + self.dropout12(src2)
        src_q = self.norm1(src_q)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src_q))))
        src_q = src_q + self.dropout2(src2)
        src_q = self.norm2(src_q)
        return src_q.permute(1, 0, 2), weight_cross, weight_self

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class GlobalLocalPrecption(nn.Module):

    def __init__(self, args, encoder_layer, num_layers, norm=None):
        super(GlobalLocalPrecption, self).__init__()

        self.args = args

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.norm = norm

    def forward(self, src_a, src_v, mask=None, src_key_padding_mask=None):
        
        #audio_output = src_a
        #visual_output = src_v

        for i in range(self.num_layers):
            src_a, weight_cross, weight_self = self.layers[i](src_a, src_v, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            #src_v = self.layers[i](src_v, src_a, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        #if self.norm:
        #    src_a = self.norm1(src_a)
        #    src_v = self.norm2(src_v)

        return src_a, weight_cross, weight_self





class GlobalHanLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(GlobalHanLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cm_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src_q, src_v, src_mask=None, src_key_padding_mask=None):

        src_q = src_q.permute(1, 0, 2)
        src_v = src_v.permute(1, 0, 2)
        src2 = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src_q = src_q + self.dropout12(src2)
        src_q = self.norm1(src_q)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src_q))))
        src_q = src_q + self.dropout2(src2)
        src_q = self.norm2(src_q)
        return src_q.permute(1, 0, 2)

class GlobalSelfAttn(nn.Module):

    def __init__(self, args, encoder_layer, num_layers, norm=None):
        super(GlobalSelfAttn, self).__init__()

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm2 = nn.LayerNorm(512)
        self.norm = norm

    def forward(self, src_v, mask=None, src_key_padding_mask=None):
        
        visual_output = src_v

        for i in range(self.num_layers):
            visual_output = self.layers[i](src_v, src_v, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            visual_output = self.norm2(visual_output)

        return visual_output


class R2SCP(nn.Module):

    def __init__(self, args, hidden_size=768):
        super(R2SCP, self).__init__()

        self.args = args
        self.num_layers = args.num_layers

        self.hidden_size = hidden_size
    
        #self.fc_a =  nn.Linear(128, hidden_size)
        #self.fc_v = nn.Linear(768, hidden_size)
        #self.fc_p = nn.Linear(768, hidden_size)
        #self.fc_word = nn.Linear(768, hidden_size)

        self.router_a = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * 1),
            out_features=1,
            drop=0.5,
        )
        self.router_t = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * 1),
            out_features=1,
            drop=0.5,
        )
        self.router_v = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * 1),
            out_features=1,
            drop=0.5,
        )

        self.relu = nn.ReLU()

        if self.args.question_encoder == "CLIP":
            self.fc_q = nn.Linear(768, hidden_size)
        else:
            self.fc_q = QstLstmEncoder(93, 512, 512, 1, 512)

        self.fc_spat_q = nn.Linear(768, hidden_size)

            
        # modules
        #self.TempSegsSelect_Module = TemporalSegmentSelection(args)
        #self.SpatRegsSelect_Module = SpatioRegionSelection(args)
        #self.AudioGuidedVisualAttn = AudioGuidedVisualAttn(args)

        #self.GlobalLocal_Module = GlobalLocalPrecption(args, 
        #                                               AVHanLayer(d_model=768, nhead=1, dim_feedforward=768), 
        #                                               num_layers=self.num_layers)
        #self.GlobalSelf_Module2 = GlobalLocalPrecption(args, 
        #                                               AVHanLayer(d_model=768, nhead=1, dim_feedforward=768), 
        #                                               num_layers=self.num_layers)
        #self.GlobalSelf_Module3 = GlobalLocalPrecption(args, 
        #                                               AVHanLayer(d_model=768, nhead=1, dim_feedforward=768), 
        #                                               num_layers=self.num_layers)

        self.expert_a = GlobalSelfAttn(args, 
                                                GlobalHanLayer(d_model=768, nhead=1, dim_feedforward=768), 
                                                num_layers=self.num_layers)
        self.expert_v = GlobalSelfAttn(args, 
                                                GlobalHanLayer(d_model=768, nhead=1, dim_feedforward=768), 
                                                num_layers=self.num_layers)
        self.expert_t = GlobalSelfAttn(args, 
                                                GlobalHanLayer(d_model=768, nhead=1, dim_feedforward=768), 
                                                num_layers=self.num_layers)
        self.guidance = GlobalLocalPrecption(args, AVHanLayer(d_model=768, nhead=1, dim_feedforward=768), 
                                                       num_layers=self.num_layers)

        self.a_in_proj = nn.Sequential(nn.Linear(128, self.hidden_size))
        self.t_in_proj = nn.Sequential(nn.Linear(768, self.hidden_size))
        self.v_in_proj = nn.Sequential(nn.Linear(768, self.hidden_size))
        self.s_in_proj = nn.Sequential(nn.Linear(768, self.hidden_size))
        self.dropout_a = nn.Dropout(0.5)
        self.dropout_t = nn.Dropout(0.5)
        self.dropout_v = nn.Dropout(0.5)
        self.dropout_s = nn.Dropout(0.5)

        #self.fc_a =  nn.Linear(128, hidden_size)
        #self.fc_v = nn.Linear(768, hidden_size)
        #self.fc_p = nn.Linear(768, hidden_size)
        #self.fc_word = nn.Linear(768, hidden_size)
        
        self.AugInformation = AugEncoder(ObjectEncoder= FMOT(d_model=768),
                                         ActionEncoder= AGE(d_model=768),
                                        max_objects= 8,
                                        visual_dim = 768,
                                        object_dim = 768,
                                        hidden_dim = 768)
        self.query_embed = nn.Embedding(8, 768)
        self.query_embed2 = nn.Embedding(8, 768)

        # fusion with audio and visual feat
        self.audio_fusion = nn.Linear(768, 768)
        self.visual_fusion = nn.Linear(768, 768)

        self.SM_fusion = nn.Linear(1024, 768)

        self.tanh_av_fusion = nn.Tanh()
        self.fc_av_fusion = nn.Linear(1024, 768)
        self.tanh_avq_fusion = nn.Tanh()

        self.Uo_v = nn.Linear(768, 768)
        self.bo_v = nn.Parameter(torch.ones(768), requires_grad=True)
        self.wo_v = nn.Linear(768, 1)

        self.Uo_a = nn.Linear(768, 768)
        self.bo_a = nn.Parameter(torch.ones(768), requires_grad=True)
        self.wo_a = nn.Linear(768, 1)

        self.linear_visual_layer = nn.Linear(2304, 768)

        self.fc_answer_pred = SpatialActivation(768)
        self.fc_answer_pred2 = SpatialActivation(768)
        self.fc_answer_pred3 = SpatialActivation(768)



    def Fusion(self, visual, object_v, object_a):
        #if fusion_object:
        U_objs = self.Uo_v(object_v)
        attn_feat = visual.unsqueeze(2) + U_objs.unsqueeze(1) + self.bo_v  # (bsz, sample_numb, max_objects, hidden_dim)
        attn_weights = self.wo_v(torch.tanh(attn_feat))  # (bsz, sample_numb, max_objects, 1)
        attn_weights = attn_weights.softmax(dim=-2)  # (bsz, sample_numb, max_objects, 1)
        attn_objects = attn_weights * attn_feat
        attn_objects = attn_objects.sum(dim=-2)  # (bsz, sample_numb, hidden_dim)

        U_objs_a = self.Uo_a(object_a)
        attn_feat_a = visual.unsqueeze(2) + U_objs_a.unsqueeze(1) + self.bo_a  # (bsz, sample_numb, max_objects, hidden_dim)
        attn_weights_a = self.wo_a(torch.tanh(attn_feat_a))  # (bsz, sample_numb, max_objects, 1)
        attn_weights_a = attn_weights_a.softmax(dim=-2)  # (bsz, sample_numb, max_objects, 1)
        attn_objects_a = attn_weights_a * attn_feat_a
        attn_objects_a = attn_objects_a.sum(dim=-2)  # (bsz, sample_numb, hidden_dim)

        features = torch.cat([visual, attn_objects, attn_objects_a], dim=-1)
        output = self.linear_visual_layer(features)
        context = torch.max(output, dim=1)[0]  # (bsz, hidden_dim)  
        return context

    def forward(self, audio, visual, patch, question, qst_word, retrival_audios_feat=None, retrival_videos_feat=None, flag=False):

        ### 1. features input 
        # audio: [B, T, C]
        # visual: [B, T, C]
        # question: [B, C]
        # patch: [B, T, N, C], N: patch numbers

        B, seq_len, C = audio.shape

        #if len(audio.size()) > 3:
        #    audio = audio.squeeze()
        #    if neg_audios_feat!=None:
        #        neg_audios_feat = neg_audios_feat.squeeze()
        #audio_feat = self.dropout_a(self.a_in_proj(audio))                   # [B, T, C]

        visual_feat = self.dropout_v(self.v_in_proj(visual))
        #if self.args.use_word:
        #    word_feat = self.dropout_t(self.t_in_proj(qst_word)).squeeze(-3)  # [B, 77, C]
        qst_feat = self.dropout_s(self.s_in_proj(question)).squeeze(-2)      # [B, C]

        word_feat = torch.randn_like(self.dropout_t(self.t_in_proj(qst_word)).squeeze(-3))

        #qst_feat = torch.randn_like(self.dropout_s(self.s_in_proj(question)).squeeze(-2))

        if flag == True:
            #visual_feat = self.dropout_v(self.v_in_proj(retrival_videos_feat))
            audio_feat = self.dropout_a(self.a_in_proj(retrival_audios_feat))

            #com_fea = self.dropout_v(self.v_in_proj(self.a_in_proj(audio)))
            com_fea = self.dropout_a(self.v_in_proj(visual))

        else:
            #visual_feat = self.dropout_v(self.v_in_proj(visual))
            #com_fea = self.dropout_v(self.v_in_proj(self.a_in_proj(audio)))
            audio_feat = self.dropout_a(self.a_in_proj(audio))
            com_fea = self.dropout_a(self.v_in_proj(visual))

        contrastive_loss = None

        #weight_a = torch.softmax(weight_a, dim=-1)
        #weight_t = torch.softmax(weight_t, dim=-1)
        #weight_v = torch.softmax(weight_v, dim=-1)
        #weight_a = weight_a.unsqueeze(-1).repeat(1, 1, 1, self.hidden_size)
        #weight_t = weight_t.unsqueeze(-1).repeat(1, 1, 1, self.hidden_size)
        #weight_v = weight_v.unsqueeze(-1).repeat(1, 1, 1, self.hidden_size)


        #query_pos_v = self.query_embed.weight
        #query_pos_a = self.query_embed2.weight
        #_, object_visual= self.AugInformation(
        #    visual = visual_feat,
        #    objects = None,
        #    query_pos = query_pos_v
        #)
        #_, object_audio= self.AugInformation(
        #    visual = audio_feat,
        #    objects = None,
        #    query_pos = query_pos_a
        #)

        #fusion_feat = torch.cat((audio_feat, visual_feat, word_feat), dim=1)

        #if flag == False or flag == True:
        if flag == True:
            #x_out_a = torch.cat([self.expert_a(audio_feat), self.expert_v(audio_feat), self.expert_t(audio_feat)], dim=-1) 
            #x_out_t = torch.cat([self.expert_a(word_feat), self.expert_v(word_feat), self.expert_t(word_feat)], dim=-1)
            #x_out_v = torch.cat([self.expert_a(visual_feat), self.expert_v(visual_feat), self.expert_t(visual_feat)], dim=-1)
            #x_unweighted_a = x_out_a.reshape(B, seq_len, 3, self.hidden_size)
            #x_unweighted_t = x_out_t.reshape(B, 77, 3, self.hidden_size)
            #x_unweighted_v = x_out_v.reshape(B, seq_len, 3, self.hidden_size)
            #x_out_a = torch.sum(weight_a * x_unweighted_a, dim=2)
            #x_out_t = torch.sum(weight_t * x_unweighted_t, dim=2)
            #x_out_v = torch.sum(weight_v * x_unweighted_v, dim=2)
            x_out_a = self.expert_a(audio_feat)
            x_out_t = self.expert_t(word_feat)
            x_out_v = self.expert_v(visual_feat)

            sim = []
            lenv = x_out_a.shape[1]
            for i in range(lenv):
                audio_frame = x_out_a[:, i, :].unsqueeze(1)
                A = torch.mean(x_out_v, dim = 1, keepdim = True)
                cos_sim2 = F.cosine_similarity(A, audio_frame, dim=-1)        # [36,64]
                sim.append(cos_sim2)
            sim = torch.stack(sim, dim=1).reshape(B, seq_len)

            _, indices_bottomk = torch.topk(-sim, k=5, dim=1)
            # 将 indices 转换为 boolean mask shape = [64,60]
            Bottom_noise = torch.zeros_like(sim, dtype=torch.bool)            # [64,60]
            # scatter_ 或者 advanced indexing 填 True
            batch_idx = torch.arange(sim.size(0), device=sim.device).unsqueeze(1).expand(-1, 5)  # [64,k]
            Bottom_noise[batch_idx, indices_bottomk] = True                 # 将被选中位置置 True

            comm_fea = self.expert_a(com_fea)
            guidance_comm_fea, weight_cross, weight_self = self.guidance(comm_fea,x_out_t)
            weight_cross = torch.sum(weight_cross, dim = -1, keepdim = False)
            weight_self = torch.sum(weight_self, dim = -1, keepdim = False)
            weight_all = weight_cross + weight_self
            _, indices = weight_all.topk(k=5, dim=1, largest=True, sorted=False)  # indices: [64, k]
            Top_comm = torch.zeros_like(weight_all, dtype=torch.bool) 
            Top_comm.scatter_(1, indices, True)
            
            for i in range(B):
                # 找出 mask 为 True 的位置索引
                idx_A = Bottom_noise[i].nonzero(as_tuple=True)[0]  # shape [5]
                idx_B = Top_comm[i].nonzero(as_tuple=True)[0]  # shape [5]
                # 替换 A 中对应的 token
                x_out_a[i, idx_A] = guidance_comm_fea[i, idx_B]

            weight_a, weight_t, weight_v = self.router_a(x_out_a.mean(dim=1)), self.router_t(x_out_t.mean(dim=1)), self.router_v(x_out_v.mean(dim=1))
            weight = torch.softmax(torch.cat([weight_a, weight_t, weight_v], dim=-1),dim=-1)
            weight_a = weight[:,0].unsqueeze(-1).unsqueeze(-1)
            weight_t = weight[:,1].unsqueeze(-1).unsqueeze(-1)
            weight_v = weight[:,2].unsqueeze(-1).unsqueeze(-1)
            x_out_a, x_out_t, x_out_v = weight_a * x_out_a, weight_t * x_out_t, weight_v * x_out_v

            avt_fusion_feat = torch.cat((x_out_a, x_out_v, x_out_t), dim=1)
            #av_fusion_feat = torch.cat([x_out_a, x_out_v], dim=-1).reshape(B, seq_len, 2, self.hidden_size)
            #av_fusion_feat = torch.sum(weight * av_fusion_feat, dim=2)
            #avt_fusion_feat = torch.cat((av_fusion_feat, x_out_t), dim=1)
        else:
            x_out_a = self.expert_a(audio_feat)
            x_out_t = self.expert_t(word_feat)
            x_out_v = self.expert_v(visual_feat)

            #weight_a, weight_t, weight_v = self.router_a(x_out_a.mean(dim=1)), self.router_t(x_out_t.mean(dim=1)), self.router_v(x_out_v.mean(dim=1))
            #weight = torch.softmax(torch.cat([weight_a, weight_t, weight_v], dim=-1),dim=-1)
            #weight_a = weight[:,0].unsqueeze(-1).unsqueeze(-1)
            #weight_t = weight[:,1].unsqueeze(-1).unsqueeze(-1)
            #weight_v = weight[:,2].unsqueeze(-1).unsqueeze(-1)
            #x_out_a, x_out_t, x_out_v = weight_a * x_out_a, weight_t * x_out_t, weight_v * x_out_v

            avt_fusion_feat = torch.cat((x_out_a, x_out_v, x_out_t), dim=1)

        #av_fusion_feat = self.Fusion(av_fusion_feat, object_visual, object_audio)
        #if neg_visual_feat != None and neg_audios_feat !=None:
        #    tau = 0.1
#
        #    neg_audio_feat = self.fc_a(neg_audios_feat)         
        #    neg_visual_feat = self.fc_v(neg_visual_feat)  
        #    #neg_fusion_feat_a = torch.cat((neg_audio_feat, word_feat), dim=1)
        #    #neg_fusion_feat_v = torch.cat((neg_visual_feat, word_feat), dim=1)  
        #    #neg_fusion_feat_a = self.GlobalSelf_Module2(neg_fusion_feat_a)
        #    #neg_fusion_feat_v = self.GlobalSelf_Module3(neg_fusion_feat_v) 
#
        #    neg_fusion_feat_a = self.GlobalSelf_Module2(neg_audio_feat, word_feat)
        #    #neg_fusion_feat_a = torch.cat((n_g_a, n_g_a_que), dim=1)
#
        #    neg_fusion_feat_v = self.GlobalSelf_Module3(neg_visual_feat, word_feat)
        #    #neg_fusion_feat_v = torch.cat((n_g_v, n_g_v_que), dim=1)
#
        #    neg_fusion_feat_a = neg_fusion_feat_a.mean(dim=-2)
        #    neg_fusion_feat_v = neg_fusion_feat_v.mean(dim=-2)
#
        #    sim_positive = cosine_similarity(av_fusion_feat, av_fusion_feat2.mean(dim=-2)) / tau + cosine_similarity(av_fusion_feat, av_fusion_feat3.mean(dim=-2)) / tau
        #    p_i_1 = torch.exp(sim_positive)
#
        #    neg_sim_negative = cosine_similarity(av_fusion_feat, neg_fusion_feat_a) / tau + cosine_similarity(av_fusion_feat, neg_fusion_feat_v) / tau
        #    p_neg_sum = torch.exp(neg_sim_negative)
#
        #    contrastive_loss = -torch.log(p_i_1 / (p_neg_sum + p_i_1))
        #    contrastive_loss = contrastive_loss.mean()
#
        #av_fusion_feat = av_fusion_feat.squeeze().unsqueeze(1)
        #av_fusion_feat2 = av_fusion_feat2.squeeze().unsqueeze(1)
        #av_fusion_feat3 = av_fusion_feat3.squeeze().unsqueeze(1)

        answer_pred = self.fc_answer_pred(avt_fusion_feat,qst_feat.squeeze().unsqueeze(1))  # [batch_size, ans_vocab_size=42]
        answer_pred2 = self.fc_answer_pred2(x_out_a,qst_feat.squeeze().unsqueeze(1))
        answer_pred3 = self.fc_answer_pred3(x_out_v,qst_feat.squeeze().unsqueeze(1))


        return answer_pred,answer_pred2,answer_pred3, _








import torch.nn.functional as F
from Model import IRT
import numpy as np
from torch import nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import logging
from matplotlib import pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from EduCDM import CDM

class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


class Net(nn.Module):
    def __init__(self, knowledge_n, exer_n, student_n,exer_emb=None):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(Net, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        if exer_emb:
            self.k_difficulty = exer_emb
        else:
            self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, input_exercise, input_knowledge_point):
        # before prednet
        # print(max(stu_id),min(stu_id))
        stu_emb = self.student_emb(stu_id)
        stat_emb = torch.sigmoid(stu_emb)
        # print("**********************************")
        # if(int(max(input_exercise))>947 or int(min(input_exercise))<1):
        # print(max(input_exercise),min(input_exercise))
        kk=self.k_difficulty(input_exercise)
        k_difficulty = torch.sigmoid(kk)
        e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise))  # * 10
        # prednet
        input_x = e_difficulty * (stat_emb - k_difficulty) * input_knowledge_point
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)


class NCDM(CDM):
    '''Neural Cognitive Diagnosis Model'''

    def __init__(self, knowledge_n, exer_n, student_n):
        super(NCDM, self).__init__()
        self.ncdm_net = Net(knowledge_n, exer_n, student_n)
    def Inv_penalty_loss(self,predict,labels):
        ids_0 = labels == 0
        ids_1 = labels == 1
        p0, p1 = predict[ids_0], predict[ids_1]
        loss0 = p0 - 0
        loss1 = 1-p1
        loss = loss0.mean() - loss1.mean()
        return loss0.mean(),-loss1.mean()
    def train(self, train_data, test_data=None, *, epoch: int, device="cpu", lr=0.001,weight=10.0,k=3,groups):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.train()
        loss_BCE = nn.BCELoss()
        trainer = torch.optim.Adam(self.ncdm_net.parameters(), lr=lr)
        best_auc = 0
        patience = 5  # 设置early stopping的patience
        no_improvement = 0  # 记录validation loss没改进的次数
        for e in range(epoch):
            losses, losses1, losses2 = [], [], []
            for batch_data in tqdm(train_data, "Epoch %s" % e):
                loss1, loss20, loss21 = [], [], []
                user_id, item_id, knowledge_emb,response = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                response: torch.Tensor = response.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                Attribute = groups[user_id].to(device)
                for env in range(k):
                    ids = Attribute == env
                    u_id = user_id[ids].to(device)
                    i_id = item_id[ids].to(device)
                    labels = response[ids].to(device)
                    num_of_1 = torch.sum(labels)
                    k_id=knowledge_emb[ids]
                    predict_labels: torch.Tensor = self.ncdm_net(u_id, i_id, k_id)
                    loss_bce = loss_BCE(predict_labels, labels)
                    inv_penalty0, inv_penalty1 = self.Inv_penalty_loss(predict_labels, labels)
                    # loss1.append(loss_bce*len(labels))
                    loss1.append(loss_bce)
                    loss20.append(inv_penalty0.mean())
                    loss21.append(inv_penalty1.mean())
                e_loss = torch.stack(loss1).sum()
                inv_loss = torch.stack(loss20).var() + torch.stack(loss21).var()
                loss = e_loss + weight * inv_loss
                # back propagation
                trainer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.irt_net.parameters(), max_norm=1.0)  # 梯度裁剪避免梯度爆炸
                trainer.step()
                losses.append(loss.mean().item())
                losses1.append(e_loss.mean().item())
                losses2.append(inv_loss.mean().item())

            if test_data is not None:
                y_true, y_pred,auc, accuracy = self.eval(test_data, device=device)
                # 如果validation accuracy提升,则更新best model
                if auc > best_auc:
                    best_auc = auc
                    bestmodel=torch.save(self.ncdm_net.state_dict(), "temp_model.snapshot")
                    no_improvement = 0
                else:
                    no_improvement += 1

                # 如果validation loss连续patience个epoch没改进,提前终止训练
                if no_improvement == patience:
                    print('Early stopping!')
                    self.ncdm_net.load_state_dict(torch.load("temp_model.snapshot"))
                    break
                print("[Epoch %d] auc: %.6f, accuracy: %.6f" % (e, auc, accuracy))

    def eval(self, test_data, device="cpu"):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.eval()
        y_true, y_pred = [], []
        for batch_data in tqdm(test_data, "Evaluating"):
            user_id, item_id, knowledge_emb, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)
            pred: torch.Tensor = self.ncdm_net(user_id, item_id, knowledge_emb)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        return y_true, y_pred,roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def save(self, filepath):
        torch.save(self.ncdm_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.ncdm_net.load_state_dict(torch.load(filepath))  # , map_location=lambda s, loc: s
        logging.info("load parameters from %s" % filepath)


def irt2pl(theta, a, b, *, F=np):
    return 1 / (1 + F.exp(- F.sum(F.multiply(a, theta), axis=-1) + b))


class MIRTNet(nn.Module):
    def __init__(self, user_num, item_num, latent_dim, a_range, irf_kwargs=None):
        super(MIRTNet, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.irf_kwargs = irf_kwargs if irf_kwargs is not None else {}
        self.theta = nn.Embedding(self.user_num, latent_dim)
        self.a = nn.Embedding(self.item_num, latent_dim)
        self.b = nn.Embedding(self.item_num, 1)
        self.a_range = a_range

    def forward(self, user, item):
        theta = torch.squeeze(self.theta(user), dim=-1)
        a = torch.squeeze(self.a(item), dim=-1)
        if self.a_range is not None:
            a = self.a_range * torch.sigmoid(a)
        else:
            a = F.softplus(a)
        b = torch.squeeze(self.b(item), dim=-1)
        if torch.max(theta != theta) or torch.max(a != a) or torch.max(b != b):  # pragma: no cover
            raise ValueError('ValueError:theta,a,b may contains nan!  The a_range is too large.')
        return self.irf(theta, a, b, **self.irf_kwargs)

    @classmethod
    def irf(cls, theta, a, b, **kwargs):
        return irt2pl(theta, a, b, F=torch)


class MIRT(CDM):
    def __init__(self, user_num, item_num, latent_dim, a_range=None):
        super(MIRT, self).__init__()
        self.irt_net = MIRTNet(user_num, item_num, latent_dim, a_range)

    def Inv_penalty_loss(self,predict,labels):
        ids_0 = labels == 0
        ids_1 = labels == 1
        p0, p1 = predict[ids_0], predict[ids_1]
        loss0 = p0 - 0
        loss1 = 1-p1
        loss = loss0.mean() - loss1.mean()
        return loss0.mean(),-loss1.mean()

    def train(self, train_data, test_data=None, *, epoch: int, device="cpu", lr=0.001,weight=10.0,k=3,groups):
        self.irt_net = self.irt_net.to(device)
        loss_BCE = nn.BCELoss()
        trainer = torch.optim.Adam(self.irt_net.parameters(), lr)
        best_auc = 0
        patience = 5  # 设置early stopping的patience
        no_improvement = 0  # 记录validation loss没改进的次数
        fair=1
        for e in range(epoch):
            losses,losses1,losses2 = [],[],[]
            for batch_data in tqdm(train_data, "Epoch %s" % e):
                loss1, loss20 ,loss21= [], [],[]
                user_id, item_id, response= batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                response: torch.Tensor = response.to(device)
                Attribute = groups[user_id].to(device)
                for env in range(k):
                    ids = Attribute == env
                    u_id=user_id[ids].to(device)
                    i_id=item_id[ids].to(device)
                    labels=response[ids].to(device)
                    num_of_1=torch.sum(labels)
                    predict_labels=self.irt_net(u_id, i_id)
                    loss_bce=loss_BCE(predict_labels,labels)
                    inv_penalty0, inv_penalty1= self.Inv_penalty_loss(predict_labels, labels)
                    # loss1.append(loss_bce*len(labels))
                    loss1.append(loss_bce)
                    loss20.append(inv_penalty0.mean())
                    loss21.append(inv_penalty1.mean())
                e_loss = torch.stack(loss1).sum()
                inv_loss = torch.stack(loss20).var()+torch.stack(loss21).var()
                loss = e_loss + weight * inv_loss
                # back propagation
                trainer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.irt_net.parameters(), max_norm=1.0)  # 梯度裁剪避免梯度爆炸
                trainer.step()
                losses.append(loss.mean().item())
                losses1.append(e_loss.mean().item())
                losses2.append(inv_loss.mean().item())

            if test_data is not None:
                y_true, y_pred, auc, accuracy = self.eval(test_data, device=device)
                # 如果validation accuracy提升,则更新best model
                if auc > best_auc:
                    best_auc = auc
                    bestmodel = torch.save(self.irt_net.state_dict(), "temp_model.snapshot")
                    no_improvement = 0
                else:
                    no_improvement += 1
                # 如果validation loss连续patience个epoch没改进,提前终止训练
                if no_improvement >= patience:
                    if np.mean(losses2) > fair or auc <= 0.79:
                        print('Early stopping!')
                        self.irt_net.load_state_dict(torch.load("temp_model.snapshot"))
                        break
                    else:
                        fair = np.mean(losses2)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f" % (e, auc, accuracy))

    def eval(self, test_data, device="cpu") -> tuple:
        self.irt_net = self.irt_net.to(device)
        self.irt_net.eval()
        y_pred = []
        y_true = []
        for batch_data in tqdm(test_data, "evaluating"):
            user_id, item_id, response,*Attribute  = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            pred: torch.Tensor = self.irt_net(user_id, item_id)
            y_pred.extend(pred.tolist())
            y_true.extend(response.tolist())

        self.irt_net.train()
        return y_pred,y_true ,roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def save(self, filepath):
        torch.save(self.irt_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.irt_net.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)


__all__ = ["irf", "irt3pl"]
def irf(theta, a, b, c, D=1.702, *, F=np):
    return c + (1 - c) / (1 + F.exp(-D * a * (theta - b)))
irt3pl = irf
class IRTNet(nn.Module):
    def __init__(self, user_num, item_num, value_range, a_range, irf_kwargs=None):
        super(IRTNet, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.irf_kwargs = irf_kwargs if irf_kwargs is not None else {}
        self.theta = nn.Embedding(self.user_num, 1)
        self.a = nn.Embedding(self.item_num, 1)
        self.b = nn.Embedding(self.item_num, 1)
        self.c = nn.Embedding(self.item_num, 1)
        self.value_range = value_range
        self.a_range = a_range

    def forward(self, user, item):
        theta = torch.squeeze(self.theta(user), dim=-1)
        a = torch.squeeze(self.a(item), dim=-1)
        b = torch.squeeze(self.b(item), dim=-1)
        c = torch.squeeze(self.c(item), dim=-1)
        c = torch.sigmoid(c)
        if self.value_range is not None:
            theta = self.value_range * (torch.sigmoid(theta) - 0.5)
            b = self.value_range * (torch.sigmoid(b) - 0.5)
        if self.a_range is not None:
            a = self.a_range * torch.sigmoid(a)
        else:
            a = F.softplus(a)
        if torch.max(theta != theta) or torch.max(a != a) or torch.max(b != b):  # pragma: no cover
            raise ValueError('ValueError:theta,a,b may contains nan!  The value_range or a_range is too large.')
        return self.irf(theta, a, b, c, **self.irf_kwargs)

    @classmethod
    def irf(cls, theta, a, b, c, **kwargs):
        return irt3pl(theta, a, b, c, F=torch, **kwargs)

class IRT(CDM):
    def __init__(self, user_num, item_num, value_range=None, a_range=None):
        super(IRT, self).__init__()
        self.irt_net = IRTNet(user_num, item_num, value_range, a_range)
    def Inv_penalty_loss(self,predict,labels):
        ids_0 = labels == 0
        ids_1 = labels == 1
        p0, p1 = predict[ids_0], predict[ids_1]
        loss0 = p0 - 0
        loss1 = 1-p1
        loss = loss0.mean() - loss1.mean()
        return loss0.mean(),-loss1.mean()
#         return loss


    def train(self, train_data, test_data=None, *, epoch: int, device="cpu", lr=0.001,weight=10.0,k=3,groups) -> ...:
        self.irt_net = self.irt_net.to(device)
        loss_BCE = nn.BCELoss()
        # loss_function=
        trainer = torch.optim.Adam(self.irt_net.parameters(), lr)
        best_auc = 0
        patience = 5  # 设置early stopping的patience
        no_improvement = 0  # 记录validation loss没改进的次数
        fair = 1
        for e in range(epoch):
            losses,losses1,losses2 = [],[],[]
            for batch_data in tqdm(train_data, "Epoch %s" % e):
                loss1, loss20 ,loss21= [], [],[]
                user_id, item_id, response= batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                response: torch.Tensor = response.to(device)
                Attribute = groups[user_id].to(device)
                for env in range(k):
                    ids = Attribute == env
                    u_id=user_id[ids].to(device)
                    i_id=item_id[ids].to(device)
                    labels=response[ids].to(device)
                    num_of_1=torch.sum(labels)
                    predict_labels=self.irt_net(u_id, i_id)
                    loss_bce=loss_BCE(predict_labels,labels)
                    inv_penalty0, inv_penalty1= self.Inv_penalty_loss(predict_labels, labels)
                    # loss1.append(loss_bce*len(labels))
                    loss1.append(loss_bce)
                    loss20.append(inv_penalty0.mean())
                    loss21.append(inv_penalty1.mean())
                e_loss = torch.stack(loss1).sum()
                inv_loss = torch.stack(loss20).var()+torch.stack(loss21).var()
                loss=e_loss+weight*inv_loss
                # back propagation
                trainer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.irt_net.parameters(), max_norm=1.0)#梯度裁剪避免梯度爆炸
                trainer.step()
                losses.append(loss.mean().item())
                losses1.append(e_loss.mean().item())
                losses2.append(inv_loss.mean().item())

            print("[Epoch %d] LogisticLoss: %.6f" % (e, float(np.mean(losses))))
            print("[Epoch %d] e_Loss: %.6f" % (e, float(np.mean(losses1))))
            print("[Epoch %d] inv_Loss: %.6f" % (e, float(np.mean(losses2))))

            if test_data is not None:
                y_true, y_pred,auc, accuracy = self.eval(test_data, device=device)
#                 如果validation accuracy提升,则更新best model
                if auc > best_auc:
                    best_auc = auc
                    bestmodel = torch.save(self.irt_net.state_dict(), "temp_model.snapshot")
                    no_improvement = 0
                else:
                    no_improvement += 1
                # 如果validation loss连续patience个epoch没改进,提前终止训练
                if no_improvement >= patience:
                    if np.mean(losses2)>fair or auc<=0.79:
                        print('Early stopping!')
                        self.irt_net.load_state_dict(torch.load("temp_model.snapshot"))
                        break
                    else:
                        fair=np.mean(losses2)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f" % (e, auc, accuracy))

    def eval(self, test_data, device="cpu") -> tuple:
        self.irt_net = self.irt_net.to(device)
        self.irt_net.eval()
        y_pred = []
        y_true = []
        for batch_data in tqdm(test_data, "evaluating"):
            user_id, item_id, response= batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            pred: torch.Tensor = self.irt_net(user_id, item_id)
            y_pred.extend(pred.tolist())
            y_true.extend(response.tolist())
        # eo_fair = self.calculate_equal_odds_and_opportunity(torch.tensor(y_true), torch.tensor(y_pred),
        #                                                     torch.tensor(attri))
        # eod, eop = eo_fair['Equal Odds'], eo_fair['Equal Opportunity']
        self.irt_net.train()
        return y_true, y_pred,roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def save(self, filepath):
        torch.save(self.irt_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.irt_net.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)
class LearnableMatrix(nn.Module):
    def __init__(self, num, k):
        super(LearnableMatrix, self).__init__()
        self.num = num
        self.k = k
        # 使用 nn.Parameter 创建可学习的矩阵
        self.matrix = nn.Parameter(torch.rand(num, k))
        self.softmax = nn.Softmax(dim=1)

    def forward(self,uid):
        # 在模型的前向传播中使用该矩阵
        matrix=self.matrix[uid]
        return self.softmax(matrix)
def Inv_penalty_loss(predict,labels,metrix):
    ids_0 = labels == 0
    ids_1 = labels == 1
    w_0,w_1=metrix[ids_0],metrix[ids_1]
    p0, p1 = predict[ids_0], predict[ids_1]
    loss0 = p0 - 0
    loss1 = 1-p1
    loss0,loss1=w_0*loss0,w_1*loss1
    loss = loss0.mean() - loss1.mean()
    return loss
# 优化过程
def optimize(modelA, modelB,train_data,device='cpu',epoch=50):
    modelA.irt_net.eval()
    modelB.train()
    # loss_BCE = nn.BCELoss()
    # optimizerA = torch.optim.Adam(modelA.parameters())
    optimizerB = torch.optim.Adam(modelB.parameters())
    for e in range(epoch):
        losses=[]
        for batch_data in tqdm(train_data, "Epoch %s" % e):
            user_id, item_id, response = batch_data
            # Attribute=modelB(user_id)
            loss1, loss2 = [], []
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            response: torch.Tensor = response.to(device)
            # Attribute: torch.Tensor = Attribute.to(device)
            matrix=modelB(user_id)
            for env in range(modelB.k):
                predict_labels = modelA.irt_net(user_id, item_id)
                env_w=matrix[:, env]
                inv_penalty = Inv_penalty_loss(predict_labels, response,env_w)
                loss2.append(inv_penalty.mean())
            # e_loss = torch.stack(loss1).sum()
            inv_loss = torch.stack(loss2).var()
            lossB=-10000*inv_loss
            optimizerB.zero_grad()
            lossB.backward()
            # print("modelB 参数的梯度:")
            # for param in modelB.parameters():
            #     print(param.grad)
            optimizerB.step()
            losses.append(lossB.item())
        print('loss:',np.mean(losses))
        # print("modelB 参数的梯度:")
        # for param in modelB.parameters():
        #     print(param.grad)

df_stu= pd.read_csv('data/PISA2018_Reading.stu.csv')
stumap=eval(open("data/stumap.map","r").read())
df_stu['stu_id:token'] = df_stu['stu_id:token'].map(stumap)
# median,mean=df_stu['ESCS:float'].quantile(1/3),df_stu['ESCS:float'].quantile(2/3)
def mergeAttribute(df,df_stu=df_stu):
    merged_df = pd.merge(df, df_stu[['stu_id:token', 'OECD:token', 'gender:token', 'ESCS:float']], on='stu_id:token', how='left')
    return merged_df
print('data reading...')
train_data = pd.read_csv("data/Pisa_train_split.csv")
valid_data = pd.read_csv("data/Pisa_val_split.csv")
test_data = pd.read_csv("data/Pisa_test_split.csv")
train_data, valid_data, test_data=mergeAttribute(train_data),mergeAttribute(valid_data),mergeAttribute(test_data)

batch_size = 2048
user_n = 1+np.max(train_data['stu_id:token'])
item_n = 1+np.max([np.max(train_data['exer_id:token']), np.max(valid_data['exer_id:token']), np.max(test_data['exer_id:token'])])
def transform(x, y, z,batch_size, **params):

    # A=A.apply(lambda x: 0 if x<-0.5 else 2 if x>0.5 else 1)
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.int64),
        torch.tensor(y, dtype=torch.int64),
        torch.tensor(z, dtype=torch.float32)
        # torch.tensor(A, dtype=torch.int64)
    )
    return DataLoader(dataset, batch_size=batch_size, **params)


# train, valid, test = [
#     transform(data["stu_id:token"], data["exer_id:token"], data["label:float"],batch_size)
#     for data in [train_data, valid_data, test_data]
# ]
def transform_ncdm(user, item, item2knowledge, knowledge_n,score, batch_size):
    knowledge_emb = torch.zeros((len(item), knowledge_n))
    for idx in range(len(item)):
        knowledge_emb[idx][np.array(item2knowledge[item[idx]])] = 1.0

    data_set = TensorDataset(
        torch.tensor(user, dtype=torch.int64),  # (1, user_n) to (0, user_n-1)
        torch.tensor(item, dtype=torch.int64),  # (1, item_n) to (0, item_n-1)
        knowledge_emb,
        torch.tensor(score, dtype=torch.float32),
    )
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)

def transform_irt(x, y, z, batch_size, **params):
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.int64),
        torch.tensor(y, dtype=torch.int64),
        torch.tensor(z, dtype=torch.float32),
        # torch.tensor(A0, dtype=torch.int64),torch.tensor(A1, dtype=torch.int64),torch.tensor(A2, dtype=torch.int64)
    )
    return DataLoader(dataset, batch_size=batch_size, **params)
def formdata(model_name,train_data, valid_data, test_data):
    if (model_name == "NCDM"):
        df_item = pd.read_csv("data/Pisa_exer.csv")
        item2knowledge = {}
        knowledge_set = set()
        for i, s in df_item.iterrows():
            item_id, knowledge_codes = s['exer_id:token'], list(set(eval(s['cpt_seq:token_seq'])))
            item2knowledge[item_id] = knowledge_codes
            knowledge_set.update(knowledge_codes)
        knowledge_n = 1 + np.max(list(knowledge_set))
        train, valid, test = [
            transform_ncdm(data["stu_id:token"], data["exer_id:token"], item2knowledge,knowledge_n, data["label:float"], batch_size)
            for data in [train_data, valid_data, test_data]
        ]
    else:
        train, valid, test = [
            transform_irt(data["stu_id:token"], data["exer_id:token"], data["label:float"], batch_size)
            for data in [train_data, valid_data, test_data]
        ]
    return train, valid, test
model_dict={'IRT':IRT(user_n,item_n),'MIRT':MIRT(user_n, item_n,latent_dim=8),'NCDM':NCDM(66, item_n, user_n)}
logging.getLogger().setLevel(logging.INFO)
model_name='IRT'
train, valid, test =formdata(model_name,train_data, valid_data, test_data)
cdm = model_dict[model_name]
# print("model loading...")
# cdm.load("irt.params")
k,weight=2,10
modelB=LearnableMatrix(user_n,k)
device='cuda' if torch.cuda.is_available() else 'cpu'
# optimize(cdm, modelB,train,device,epoch=50)
# torch.save(modelB.state_dict(), 'modelB'+str(modelB.k)+'.params')
modelB.load_state_dict(torch.load('modelparams/irt_groups_var_k2.params'))
# modelB.load_state_dict(torch.load(model_name+'_groups_var_k2.params'))
group_probs = F.softmax(modelB.matrix,dim=1)
_, predicted_groups = torch.max(group_probs, 1)
print(model_name+" training...")
cdm.train(train, valid, epoch=50, device=device,lr=0.001,weight=weight,k=k,groups=predicted_groups)
cdm.save('modelparams/'+model_name+'_rex_v2_2_w'+str(weight)+'_k'+str(k)+'.params')
# cdm.load("irt.params")
y_true, y_pred,auc, accuracy = cdm.eval(test)
print("auc: %.6f, accuracy: %.6f" % (auc, accuracy))
print('ok')

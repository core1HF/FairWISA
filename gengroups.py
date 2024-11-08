

import torch.nn.functional as F
from Model import IRT,NCDM,MIRT
import numpy as np
from torch import nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import logging
from matplotlib import pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

class LearnableMatrix(nn.Module):
    def __init__(self, num, k):
        super(LearnableMatrix, self).__init__()
        self.num = num
        self.k = k
        self.matrix = nn.Parameter(torch.randn(num, k))
        self.softmax = nn.Softmax(dim=1)

    def initialize_matrix(self):
        self.matrix.data[:, 0] = torch.ones(self.matrix.size(0))
    def forward(self,uid):
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
    num0=w_0.sum()
    num1 = w_1.sum()
    loss = loss0.sum()/num0 - loss1.sum()/num1
    return loss0.sum()/num0,-loss1.sum()/num1

def real_Inv_penalty_loss(predict,labels):
    ids_0 = labels == 0
    ids_1 = labels == 1
    p0, p1 = predict[ids_0], predict[ids_1]
    loss0 = p0 - 0
    loss1 = 1-p1
    loss = loss0.mean() - loss1.mean()
    return loss

def optimize(modelA, modelB,data,device='cpu',epoch=50):
    modelA.irt_net.to(device)
    modelB.to(device)
    modelA.irt_net.eval()
    modelB.train()
    # loss_BCE = nn.BCELoss()
    # optimizerA = torch.optim.Adam(modelA.parameters())
    optimizerB = torch.optim.Adam(modelB.parameters(),lr=0.05)
    ratio_list=[]
    for e in range(epoch):
        losses=[]
        losses2=[[]]*modelB.k
        user_id, item_id, response1 = torch.tensor(data["stu_id:token"].values, dtype=torch.int64), torch.tensor(data["exer_id:token"].values, dtype=torch.int64), torch.tensor(data["label:float"].values, dtype=torch.float32)
        # Attribute=modelB(user_id)
        loss1, loss20,loss21 = [], [],[]
        loss2=[]
        realinvloss=[]
        user_id: torch.Tensor = user_id.to(device)
        item_id: torch.Tensor = item_id.to(device)
        response: torch.Tensor = response1.to(device)
        predict_labels = modelA.irt_net(user_id, item_id)
        # Attribute: torch.Tensor = Attribute.to(device)
        matrix=modelB(user_id)
        groups=F.gumbel_softmax(matrix, tau=0.1, hard=True)
        numenv1=torch.sum(groups[:, 1]).float()
        ratio=numenv1/len(matrix)
        ratioloss = torch.abs(ratio - 0.47)
        # ratio_list.append(ratio.item())
        # if ratio<=0.5:
        #     drawratio(e+1,ratio_list)
        #     break
        for env in range(modelB.k):
            env_w=groups[:, env]
            inv_penalty0,inv_penalty1 = Inv_penalty_loss(predict_labels, response,env_w)
            loss20.append(inv_penalty0.mean())
            loss21.append(inv_penalty1.mean())
            loss2.append(inv_penalty0 + inv_penalty1)

        inv_loss = torch.stack(loss20).var()+torch.stack(loss21).var()
        # inv_loss = loss20[0].mean() - loss21[0].mean()+loss20[1].mean() - loss21[1].mean()
        # inv_loss = loss2[0]-loss2[1]+ratioloss
        # losses2[0]=loss2[0].item()
        # losses2[1]=loss2[1].item()
        # losses2[2]=loss2[2].item()
        # losses2[3].append(loss2[3].item())
        lossB=-inv_loss
        optimizerB.zero_grad()
        lossB.backward()
        optimizerB.step()
        losses.append(lossB.item())
        print('\n epoch:',e,'\nvarloss:',np.mean(losses))
        print(ratio)
        for i in range(modelB.k):
            print('loss'+str(i)+':',loss2[i].item(),end='  ')
def optimize_ncdm(modelA, modelB,data,device='cpu',epoch=50):
    modelA.ncdm_net.to(device)
    modelB.to(device)
    modelA.ncdm_net.eval()
    modelB.train()
    # loss_BCE = nn.BCELoss()
    # optimizerA = torch.optim.Adam(modelA.parameters())
    optimizerB = torch.optim.Adam(modelB.parameters(),lr=0.05)
    for e in range(epoch):
        losses,ratio_list= [],[]
        losses2 = [[]] * modelB.k
        for batch_data in tqdm(data, "Epoch %s" % e):

            user_id, item_id,  knowledge_emb,response1 = batch_data
            loss1, loss20,loss21 = [], [],[]
            loss2=[]
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            response: torch.Tensor = response1.to(device)
            knowledge_emb=knowledge_emb.to(device)
            predict_labels = modelA.ncdm_net(user_id, item_id,knowledge_emb)
            # Attribute: torch.Tensor = Attribute.to(device)
            matrix=modelB(user_id)
            groups=F.gumbel_softmax(matrix, tau=0.1, hard=True)
            numenv1=torch.sum(groups[:, 1]).float()
            ratio=numenv1/len(matrix)
            ratio_list.append(ratio)
            ratioloss = torch.abs(ratio - 0.47)
            # ratio_list.append(ratio.item())
            # if ratio<=0.5:
            #     drawratio(e+1,ratio_list)
            #     break
            for env in range(modelB.k):
                env_w=groups[:, env]
                inv_penalty0,inv_penalty1 = Inv_penalty_loss(predict_labels, response,env_w)
                loss20.append(inv_penalty0.mean())
                loss21.append(inv_penalty1.mean())
                losses2[env].append(inv_penalty0.item() + inv_penalty1.item())
                # loss2.append(inv_penalty0 + inv_penalty1)

            inv_loss = torch.stack(loss20).var()+torch.stack(loss21).var()
            # inv_loss = loss20[0].mean() - loss21[0].mean()+loss20[1].mean() - loss21[1].mean()
            lossB=-inv_loss
            optimizerB.zero_grad()
            lossB.backward()
            optimizerB.step()
            losses.append(lossB.item())
        print('\n epoch:',e,'\nvarloss:',np.mean(losses))
        print(sum(ratio_list)/len(ratio_list))
        for i in range(modelB.k):
            print('loss'+str(i)+':',np.mean(losses2[i]),end='  ')


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

batch_size = 32768
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

def transform_ncdm(user, item, item2knowledge, knowledge_n, score,batch_size):
    knowledge_emb = torch.zeros((len(item), knowledge_n))
    for idx in range(len(item)):
        knowledge_emb[idx][np.array(item2knowledge[item[idx]])] = 1.0

    data_set = TensorDataset(
        torch.tensor(user, dtype=torch.int64),  # (1, user_n) to (0, user_n-1)
        torch.tensor(item, dtype=torch.int64),  # (1, item_n) to (0, item_n-1)
        knowledge_emb,
        torch.tensor(score, dtype=torch.float32)
    )
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)
# train, valid, test = [
#     transform(data["stu_id:token"], data["exer_id:token"], data["label:float"],batch_size)
#     for data in [train_data, valid_data, test_data]
# ]

model_name='IRT'
model_dict={'IRT':IRT(user_n,item_n),'MIRT':MIRT(user_n, item_n,latent_dim=8),'NCDM': NCDM(66, item_n, user_n)}
cdm = model_dict[model_name]
print(model_name+" model loading...")

logging.getLogger().setLevel(logging.INFO)
k=2
modelB=LearnableMatrix(user_n,k=k)
device='cuda' if torch.cuda.is_available() else 'cpu'
if model_name=='NCDM':
    cdm.load('modelparams/NCDM_cpu_pisa.snapshot')
    df_item = pd.read_csv("data/Pisa_exer.csv")
    item=df_item['exer_id:token']
    item2knowledge = {}
    knowledge_set = set()
    for i, s in df_item.iterrows():
        item_id, knowledge_codes = s['exer_id:token'], list(set(eval(s['cpt_seq:token_seq'])))
        item2knowledge[item_id] = knowledge_codes
        knowledge_set.update(knowledge_codes)
    knowledge_n = 1 + np.max(list(knowledge_set))
    # train = transform_ncdm(valid_data["stu_id:token"], valid_data["exer_id:token"], item2knowledge, knowledge_n,
    #                        valid_data["label:float"], batch_size)
    train=transform_ncdm(train_data["stu_id:token"], train_data["exer_id:token"], item2knowledge, knowledge_n,train_data["label:float"],batch_size)
    optimize_ncdm(cdm, modelB,train,device,epoch=300)
elif model_name=='IRT':
    cdm.load("saver/irt.params")
    optimize(cdm, modelB, train_data, device, epoch=400)
else:
    cdm.load("saver/MIRT.params")
    optimize(cdm, modelB, train_data, device, epoch=400)
torch.save(modelB.state_dict(), model_name+'_groups_var_k'+str(k)+'.params')
print('ok')

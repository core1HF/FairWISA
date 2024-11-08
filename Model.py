from EduCDM import MIRT
import logging
import numpy as np
import torch
from EduCDM import CDM
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import logging
from matplotlib import pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

def calculate_accuracy(group):
    total_rows = len(group)
    true_accuracy = group['label'].sum() / total_rows if total_rows > 0 else 0
    pred_accuracy = group['pred'].sum() / total_rows if total_rows > 0 else 0
    return pd.Series({'true_accuracy': true_accuracy, 'pred_accuracy': pred_accuracy, 'A': group['A'].iloc[0]})

def calculate_FCD(stu_tensor,label_tensor,pred_tensor,A_tensor):
    pred_tensor= (pred_tensor >= 0.5).int()
    df = pd.DataFrame({
        'stu': stu_tensor,
        'pred': pred_tensor,
        'label': label_tensor,
        'A': A_tensor
    })
    # Group by 'stu', calculate accuracy, and reset index
    accuracy_df = df.groupby('stu').apply(calculate_accuracy).reset_index()
    # Drop duplicates based on 'stu'
    df = accuracy_df.drop_duplicates(subset='stu')
    # 计算 true_accuracy - pred_accuracy
    df['accuracy_diff'] = df['pred_accuracy']-df['true_accuracy']
    # 按组（A列）进行分组并计算平均值
    grouped_mean = df.groupby('A')['accuracy_diff'].mean().reset_index()
    # Print the final result_iid
    FCD = grouped_mean.loc[1, 'accuracy_diff'] - grouped_mean.loc[0, 'accuracy_diff']
    # print(grouped_mean)
    return FCD

def calculate_equal_odds_and_opportunity(y_true, y_pred, sensitive_feature):
    subgroups = set(sensitive_feature.tolist())
    equal_odds = {}
    equal_opportunity = {}
    dp = {}
    fairCD={}
    equal_acc={}
    neo={}
    for subgroup in subgroups:
      
        subgroup_idx = sensitive_feature == subgroup
        
        subgroup_y_pred = y_pred[subgroup_idx]
        subgroup_y_true = y_true[subgroup_idx]
        
        positive_prob = sum(subgroup_y_pred >= 0.5) / len(subgroup_y_pred)
        true_prob=sum(subgroup_y_true ==1) / len(subgroup_y_true)
        fairCD[subgroup]=positive_prob-true_prob
        dp[subgroup] = positive_prob
        # 计算 True Positive、False Positive、True Negative、False Negative
        tp = sum((subgroup_y_pred >= 0.5) & (subgroup_y_true == 1))
        fp = sum((subgroup_y_pred >= 0.5) & (subgroup_y_true == 0))
        tn = sum((subgroup_y_pred < 0.5) & (subgroup_y_true == 0))
        fn = sum((subgroup_y_pred < 0.5) & (subgroup_y_true == 1))
        neo[subgroup]={'TNR':tn/(tn+fp)}
        equal_odds[subgroup] = {
            'TPR': tp / (tp + fn),
            'FPR': fp / (tn + fp),
        }
        equal_acc[subgroup] = {
            'accuracy': (tp + tn) / (tp + tn + fp + fn)
        }
        equal_opportunity[subgroup] = {
            'TPR': tp / (tp + fn),
        }
    dp['sub'],equal_opportunity['sub'],fairCD['sub'] = dp[1]-dp[0],equal_opportunity[1]['TPR']-equal_opportunity[0]['TPR'],fairCD[1]-fairCD[0]
    neo['sub'],equal_odds['sub'],equal_acc['sub'] = neo[1]['TNR']-neo[0]['TNR'],equal_odds[1]['FPR'] - equal_odds[0]['FPR'],equal_acc[1]['accuracy']-equal_acc[0]['accuracy']
    return {
        'dp': dp,
        'FairCD': fairCD,
        'Equal Opportunity': equal_opportunity,
        'Equal Odds': equal_odds,
        'Equal Acc':equal_acc,
        'NEO':neo
    }

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

    def train(self, train_data, test_data=None, epoch=10, device="cpu", lr=0.002, silence=False):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.train()
        loss_function = nn.BCELoss()
        optimizer = torch.optim.Adam(self.ncdm_net.parameters(), lr=lr)
        best_auc = 0
        patience = 5  
        no_improvement = 0  
        for epoch_i in range(epoch):
            # print(epoch_i)
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i):
                batch_count += 1
                user_id, item_id, knowledge_emb, y = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                y: torch.Tensor = y.to(device)
                pred: torch.Tensor = self.ncdm_net(user_id, item_id, knowledge_emb)
                loss = loss_function(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())
                # print(loss)
            print("[Epoch %d] average loss: %.6f" % (epoch_i, float(np.mean(epoch_losses))))
            if test_data is not None:
                y_true, y_pred,auc, accuracy = self.eval(test_data, device=device)
                if auc > best_auc:
                    best_auc = auc
                    bestmodel=torch.save(self.ncdm_net.state_dict(), "temp_model.snapshot")
                    no_improvement = 0
                else:
                    no_improvement += 1
                if no_improvement == patience:
                    print('Early stopping!')
                    self.ncdm_net.load_state_dict(torch.load("temp_model.snapshot"))
                    break
                print("[Epoch %d] auc: %.6f, accuracy: %.6f" % (epoch_i, auc, accuracy))

    def eval(self, test_data, device="cpu"):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.eval()
        y_true, y_pred,uids = [], [],[]
        attri0,attri1,attri2=[],[],[]
        for batch_data in tqdm(test_data, "Evaluating"):
            user_id, item_id, knowledge_emb, y, *Attribute = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)
            pred: torch.Tensor = self.ncdm_net(user_id, item_id, knowledge_emb)
            response=y.to(device)
            y_pred.extend(pred.tolist())
            y_true.extend(response.tolist())
            uids.extend(user_id.tolist())
            attri0.extend(Attribute[0].tolist())
            attri1.extend(Attribute[1].tolist())
            attri2.extend(Attribute[2].tolist())
        for attri in [attri0, attri1, attri2]:
            FCD = calculate_FCD(torch.tensor(uids), torch.tensor(y_true), torch.tensor(y_pred), torch.tensor(attri))
            eo_fair = calculate_equal_odds_and_opportunity(torch.tensor(y_true), torch.tensor(y_pred),
                                                           torch.tensor(attri))
            dp, eop, fairCD, eod, e_acc, neo = eo_fair['dp'], eo_fair['Equal Opportunity'], eo_fair['FairCD'], eo_fair[
                'Equal Odds'], eo_fair['Equal Acc'], eo_fair['NEO']
            print(dp, eop, fairCD, eod, e_acc, neo, "FCD:", FCD)
        return y_true, y_pred, roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def save(self, filepath):
        torch.save(self.ncdm_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.ncdm_net.load_state_dict(torch.load(filepath))  # , map_location=lambda s, loc: s
        logging.info("load parameters from %s" % filepath)

def irt2pl(theta, a, b, *, F=np):
    """

    Parameters
    ----------
    theta
    a
    b
    F

    Returns
    -------

    Examples
    --------
    >>> theta = [1, 0.5, 0.3]
    >>> a = [-3, 1, 3]
    >>> b = 0.5
    >>> irt2pl(theta, a, b) # doctest: +ELLIPSIS
    0.109...
    >>> theta = [[1, 0.5, 0.3], [2, 1, 0]]
    >>> a = [[-3, 1, 3], [-3, 1, 3]]
    >>> b = [0.5, 0.5]
    >>> irt2pl(theta, a, b) # doctest: +ELLIPSIS
    array([0.109..., 0.004...])
    """
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

    def train(self, train_data, test_data=None, *, epoch: int, device="cpu", lr=0.001,weight=10.0,rex=False) -> ...:
        self.irt_net = self.irt_net.to(device)

        loss_function = nn.BCELoss()
        trainer = torch.optim.Adam(self.irt_net.parameters(), lr)

        for e in range(epoch):
            losses = []
            for batch_data in tqdm(train_data, "Epoch %s" % e):
                user_id, item_id, response = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                predicted_response: torch.Tensor = self.irt_net(user_id, item_id)
                response: torch.Tensor = response.to(device)
                loss = loss_function(predicted_response, response)
                # back propagation
                trainer.zero_grad()
                loss.backward()
                trainer.step()

                losses.append(loss.mean().item())
            print("[Epoch %d] LogisticLoss: %.6f" % (e, float(np.mean(losses))))
            if test_data is not None:
                auc, accuracy = self.eval(test_data, device=device)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f" % (e, auc, accuracy))

    def eval(self, test_data, device="cpu") -> tuple:
        self.irt_net = self.irt_net.to(device)
        self.irt_net.eval()
        y_pred = []
        y_true = []
        uids=[]
        attri0,attri1,attri2= [],[],[]
        for batch_data in tqdm(test_data, "evaluating"):
            user_id, item_id, response,*Attribute = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            pred: torch.Tensor = self.irt_net(user_id, item_id)
            y_pred.extend(pred.tolist())
            y_true.extend(response.tolist())
            uids.extend(user_id.tolist())
            attri0.extend(Attribute[0].tolist())
            attri1.extend(Attribute[1].tolist())
            attri2.extend(Attribute[2].tolist())
        for attri in [attri0, attri1, attri2]:
            FCD = calculate_FCD(torch.tensor(uids), torch.tensor(y_true), torch.tensor(y_pred), torch.tensor(attri))
            eo_fair = calculate_equal_odds_and_opportunity(torch.tensor(y_true), torch.tensor(y_pred),
                                                           torch.tensor(attri))
            dp, eop, fairCD, eod, e_acc, neo = eo_fair['dp'], eo_fair['Equal Opportunity'], eo_fair['FairCD'], eo_fair[
                'Equal Odds'], eo_fair['Equal Acc'], eo_fair['NEO']
            print(dp, eop, fairCD, eod, e_acc, neo, "FCD:", FCD)
        self.irt_net.train()
        return y_true, y_pred,roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

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

    def eval(self, test_data, device="cpu") -> tuple:
        self.irt_net = self.irt_net.to(device)
        self.irt_net.eval()
        uids=[]
        y_pred = []
        y_true = []
        attri0,attri1,attri2= [],[],[]
        for batch_data in tqdm(test_data, "evaluating"):
            user_id, item_id, response,*Attribute = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            pred: torch.Tensor = self.irt_net(user_id, item_id)
            y_pred.extend(pred.tolist())
            y_true.extend(response.tolist())
            uids.extend(user_id.tolist())
            attri0.extend(Attribute[0].tolist())
            attri1.extend(Attribute[1].tolist())
            attri2.extend(Attribute[2].tolist())
        for attri in [attri0,attri1,attri2]:
            FCD=calculate_FCD(torch.tensor(uids),torch.tensor(y_true),torch.tensor(y_pred),torch.tensor(attri))
            eo_fair=calculate_equal_odds_and_opportunity(torch.tensor(y_true),torch.tensor(y_pred),torch.tensor(attri))
            dp,eop,fairCD,eod,e_acc,neo=eo_fair['dp'],eo_fair['Equal Opportunity'],eo_fair['FairCD'],eo_fair['Equal Odds'],eo_fair['Equal Acc'],eo_fair['NEO']
            print(dp,eop,fairCD,eod,e_acc,neo,"FCD:",FCD)
        self.irt_net.train()
        return y_true, y_pred,roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def save(self, filepath):
        torch.save(self.irt_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.irt_net.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)

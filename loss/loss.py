import torch
import torch.nn as nn
from itertools import permutations
from scipy.optimize import linear_sum_assignment 


# modified for padding cases
class BasicLoss():
    ''' The basic class for pit, fastpit and optm loss.''' 
    def __init__(self):
        self.criterion = nn.BCELoss()

    def __call__(self, preds, labels):
        '''
        args:
            preds   - B * T * n_class, float
            labels  - B * T * n_class, int
        '''
        losses = []
        assigned_labels = []
        nframe = 0
        for pred, label in zip(preds, labels):
            loss, assigned_label = self.minimize_loss(pred, label)
            losses.append(loss * len(pred))
            nframe += len(pred)
            assigned_labels.append(assigned_label)

        loss = torch.stack(losses).sum() / nframe
        return loss, assigned_labels



class PITLoss(BasicLoss):
    ''' Minimize the loss by permutation of all possible labels.'''
    def minimize_loss(self, pred, label):
        n_class = label.shape[-1]
        loss = None
        assigned_label = None
        for p in permutations(range(n_class)):
            bce_loss = self.criterion(pred, label[:, p])
            if loss is None or loss > bce_loss:
                loss = bce_loss
                assigned_label = label[:, p]
        return loss, assigned_label



class FastPITLoss(BasicLoss):
    def __init__(self):
        self.criterion = nn.BCELoss(reduction='none')

    ''' Minimize the loss by pre-construction of loss matrix and permutation.'''
    def minimize_loss(self, pred, label):
        n_class = label.shape[-1]
        # The faster construction of loss matrix, but 
        # increase space complexity (T * n_class --> T * n_class ** 2).
        ext_pred = pred.unsqueeze(2).repeat(1, 1, n_class)
        ext_label = label.unsqueeze(1).repeat(1, n_class, 1)
        loss_mat = self.criterion(ext_pred, ext_label)
        loss_mat = loss_mat.mean(dim=0)
        
        loss = None
        assigned_label = None
        for p in permutations(range(n_class)):
            bce_loss = loss_mat[range(n_class), p].mean()
            if loss is None or loss > bce_loss:
                loss = bce_loss
                assigned_label = label[:, p]
        return loss, assigned_label



class OPTMLoss(BasicLoss):
    def __init__(self):
        self.criterion = nn.BCELoss(reduction='none')

    ''' Minimize the loss by pre-construction of loss matrix and Hungarian algorithm.'''
    def minimize_loss(self, pred, label):
        n_class = label.shape[-1]
        ext_pred = pred.unsqueeze(2).repeat(1, 1, n_class)
        ext_label = label.unsqueeze(1).repeat(1, n_class, 1)
        loss_mat = self.criterion(ext_pred, ext_label)
        loss_mat = loss_mat.mean(dim=0)

        # row_ind is sorted like: [0, 1, 2, 3, ...]
        row_ind, col_ind = linear_sum_assignment(loss_mat.detach().cpu().numpy())
        loss = loss_mat[row_ind, col_ind].mean()
        assigned_label = label[:, col_ind]
        return loss, assigned_label



if __name__ == '__main__':
    B, T, n_class = 1, 4, 4
    device = 'cpu'
    pred = torch.rand(B, T, n_class).to(device)
    label = torch.randint(0, 2, (B, T, n_class)).float().to(device)

    criterion1 = PITLoss()
    criterion2 = FastPITLoss()
    criterion3 = OPTMLoss()
    loss1, assigned_label1 = criterion1(pred, label)
    loss2, assigned_label2 = criterion2(pred, label)
    loss3, assigned_label3 = criterion3(pred, label)
    print(loss1, loss2, loss3)
    print(assigned_label1)
    print(assigned_label2)
    print(assigned_label3)

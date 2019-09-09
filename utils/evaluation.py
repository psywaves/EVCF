from __future__ import print_function

import torch
from torch.autograd import Variable

import numpy as np

import time

import os
import bottleneck as bn
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# ======================================================================================================================
def evaluate_vae(args, model, train_loader, data_loader, epoch, dir, mode):
    # set loss to 0
    evaluate_loss = 0
    evaluate_re = 0
    evaluate_kl = 0

    ndcg_dist = torch.tensor([], dtype=torch.float)
    if mode == 'test':
        ndcg_20 = torch.tensor([], dtype=torch.float)
        ndcg_10 = torch.tensor([], dtype=torch.float)
        recall_50 = torch.tensor([], dtype=torch.float)
        recall_20 = torch.tensor([], dtype=torch.float)
        recall_10 = torch.tensor([], dtype=torch.float)
        recall_5 = torch.tensor([], dtype=torch.float)
        recall_1 = torch.tensor([], dtype=torch.float)

    # set model to evaluation mode
    model.eval()

    # Functions for Evaluation

    def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100):
        '''
        normalized discounted cumulative gain@k for binary relevance
        ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
        '''
        batch_users = X_pred.shape[0]
        idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
        topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                           idx_topk_part[:, :k]]
        idx_part = np.argsort(-topk_part, axis=1)
        # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
        # topk predicted score
        idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
        # build the discount template
        tp = 1. / np.log2(np.arange(2, k + 2))
        tp = torch.tensor(tp, dtype=torch.float)  # ! in order to do operations with torch tensor

        DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                             idx_topk].cpu() * tp).sum(dim=1)
        IDCG = torch.tensor([(tp[:min(n, k)]).sum()
                             for n in (heldout_batch != 0).sum(dim=1)])
        return DCG / IDCG

    def Recall_at_k_batch(X_pred, heldout_batch, k=100):
        batch_users = X_pred.shape[0]

        idx = bn.argpartition(-X_pred, k, axis=1)
        X_pred_binary = np.zeros_like(X_pred, dtype=bool)
        X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

        X_true_binary = torch.tensor((heldout_batch > 0), dtype=torch.float)
        tmp = torch.tensor(np.logical_and(X_true_binary, X_pred_binary), dtype=torch.float).sum(dim=1)
        recall = tmp / np.minimum(k, X_true_binary.sum(dim=1))
        return recall

    # evaluate
    for batch_idx, (train, test) in enumerate(data_loader):
        if args.cuda:
            train, test = train.cuda(), test.cuda()
        train, test = Variable(train), Variable(test) #! volatile deprecated

        x = train

        with torch.no_grad():
            # calculate loss function
            loss, RE, KL = model.calculate_loss(x, average=True)

            evaluate_loss += loss.data.item()
            evaluate_re += -RE.data.item()
            evaluate_kl += KL.data.item()

            # Calculate NDCG & Recall
            pred_val = model.reconstruct_x(x)
            # should be removed if not necessary
            pred_val = np.array(pred_val)
            x = np.array(x)
            pred_val[x.nonzero()] = -np.inf

            ndcg_dist = torch.cat([ndcg_dist, NDCG_binary_at_k_batch(pred_val, test, k=100)])

            if mode == 'test':
                ndcg_20 = torch.cat([ndcg_20, NDCG_binary_at_k_batch(pred_val, test, k=20)])
                ndcg_10 = torch.cat([ndcg_10, NDCG_binary_at_k_batch(pred_val, test, k=10)])
                recall_50 = torch.cat([recall_50, Recall_at_k_batch(pred_val, test, k=50)])
                recall_20 = torch.cat([recall_20, Recall_at_k_batch(pred_val, test, k=20)])
                recall_10 = torch.cat([recall_10, Recall_at_k_batch(pred_val, test, k=10)])
                recall_5 = torch.cat([recall_5, Recall_at_k_batch(pred_val, test, k=5)])
                recall_1 = torch.cat([recall_1, Recall_at_k_batch(pred_val, test, k=1)])



    # calculate final loss
    evaluate_loss /= len(data_loader)  # loss function already averages over batch size
    evaluate_re /= len(data_loader)  # re already averages over batch size
    evaluate_kl /= len(data_loader)  # kl already averages over batch size

    evaluate_ndcg = ndcg_dist.mean().data.item()

    if mode == 'test':
        eval_ndcg100 = "{:.5f}({:.4f})".format(evaluate_ndcg, ndcg_dist.std().data.item()/np.sqrt(len(ndcg_dist)))
        eval_ndcg20 = "{:.5f}({:.4f})".format(ndcg_20.mean().data.item(),ndcg_20.std().data.item()/np.sqrt(len(ndcg_20)))
        eval_ndcg10 = "{:.5f}({:.4f})".format(ndcg_10.mean().data.item(),ndcg_10.std().data.item()/np.sqrt(len(ndcg_10)))
        eval_recall50 = "{:.5f}({:.4f})".format(recall_50.mean().data.item(),recall_50.std().data.item()/np.sqrt(len(recall_50)))
        eval_recall20 = "{:.5f}({:.4f})".format(recall_20.mean().data.item(),recall_20.std().data.item()/np.sqrt(len(recall_20)))
        eval_recall10 = "{:.5f}({:.4f})".format(recall_10.mean().data.item(),recall_10.std().data.item()/np.sqrt(len(recall_10)))
        eval_recall5 = "{:.5f}({:.4f})".format(recall_5.mean().data.item(),recall_5.std().data.item()/np.sqrt(len(recall_5)))
        eval_recall1 = "{:.5f}({:.4f})".format(recall_1.mean().data.item(),recall_1.std().data.item()/np.sqrt(len(recall_1)))



    if mode == 'test':
        return evaluate_loss, evaluate_re, evaluate_kl, eval_ndcg100, \
               eval_ndcg20, eval_ndcg10, eval_recall50, eval_recall20, eval_recall10, eval_recall5, eval_recall1
    else:
        return evaluate_loss, evaluate_re, evaluate_kl, evaluate_ndcg

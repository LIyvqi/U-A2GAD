import torch
import torch.nn.functional as F
import time
import numpy
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score, confusion_matrix,precision_recall_curve,auc
from sklearn.model_selection import train_test_split
import numpy as np
from unKnn import neg_knn_graph
import dgl
from dgl.nn.pytorch.factory import KNNGraph


def train(model, g, args):
    features = g.ndata['feature']
    labels = g.ndata['label']

    index = list(range(len(labels)))
    dataset_name = args.dataset
    if dataset_name == 'amazon':
        index = list(range(3305, len(labels)))

    idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index],
                                                            train_size=args.train_ratio,
                                                            random_state=2, shuffle=True)
    idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                            test_size=0.67,
                                                            random_state=2, shuffle=True)
    train_mask = torch.zeros([len(labels)]).bool()
    val_mask = torch.zeros([len(labels)]).bool()
    test_mask = torch.zeros([len(labels)]).bool()

    train_mask[idx_train] = 1
    val_mask[idx_valid] = 1
    test_mask[idx_test] = 1

    print('train/dev/test samples: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr , weight_decay=args.wd)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wd)


    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=50)

    best_vauc_pr, best_f1, final_tf1, final_trec, final_tpre, final_tmf1, final_tauc, final_tauc_pr,best_vauc = 0., 0., 0., 0., 0., 0., 0.,0.,0.

    weight = (1-labels[train_mask]).sum().item() / labels[train_mask].sum().item()


    print('cross entropy weight: ', weight)
    time_start = time.time()
    for e in range(args.epoch):
        model.train()
        logits,_ = model(features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=torch.tensor([1., weight]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()

        logits = logits.detach()
        _ = _.detach()
        loss = loss.detach()
        features = features.detach()
        emb = None
        with torch.no_grad():
            logits, emb = model(features)
            probs = logits.softmax(1)
            f1, thres = get_best_f1(labels[val_mask], probs[val_mask])
            preds = numpy.zeros_like(labels)
            preds[probs[:, 1] > thres] = 1
            trec = recall_score(labels[test_mask], preds[test_mask])
            tpre = precision_score(labels[test_mask], preds[test_mask])
            tmf1 = f1_score(labels[test_mask], preds[test_mask], average='macro')
            tauc = roc_auc_score(labels[test_mask], probs[test_mask][:, 1].detach().numpy())

            precision, recall, _ = precision_recall_curve(labels[test_mask], probs[test_mask][:, 1].detach())
            tauc_pr = auc(recall, precision)

            vprecision, vrecall, _ = precision_recall_curve(labels[val_mask], probs[val_mask][:, 1].detach())
            vauc_pr = auc(vrecall, vprecision)

            vauc = roc_auc_score(labels[val_mask], probs[val_mask][:, 1].detach().numpy())

        if (best_vauc + best_vauc_pr + best_f1) < (f1 + vauc + vauc_pr):
            kg = KNNGraph(args.topk)
            emb = emb.detach()
            knng = kg(emb)
            knng = dgl.to_bidirected(knng)
            knng.ndata['feature'] = model.g.ndata['feature']
            model.knng = knng

            neg_g = neg_knn_graph(emb, args.neg_topk, exclude_self=True)
            neg_g = dgl.to_bidirected(neg_g)
            neg_g.ndata['feature'] = model.g.ndata['feature']
            model.neg_knng = neg_g

            best_vauc_pr = vauc_pr
            best_f1 = f1
            best_vauc = vauc
            final_trec = trec
            final_tpre = tpre
            final_tmf1 = tmf1
            final_tauc = tauc
            final_tauc_pr = tauc_pr
            print(e,'== Test: REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f} AUCpr {:.2f}'.format(final_trec*100,
                                                                     final_tpre*100, final_tmf1*100, final_tauc*100, final_tauc_pr*100))

    time_end = time.time()
    print('time cost: ', time_end - time_start, 's')
    print('Test: REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f}  AUCpr {:.2f}'.format(final_trec*100,
                                                                     final_tpre*100, final_tmf1*100, final_tauc*100, final_tauc_pr*100))
    return final_tmf1, final_tauc, final_tauc_pr


# threshold adjusting for best macro f1
def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:,1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre
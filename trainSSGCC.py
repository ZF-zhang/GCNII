from __future__ import division
from __future__ import print_function
from doctest import OutputChecker
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from utils import *
from modelSSGCC import *
import uuid
from arguments import args
from sklearn.metrics import f1_score, precision_score, recall_score



# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=24, help='Random seed.')
parser.add_argument('--epochs', type=int, default=3000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate.')
parser.add_argument('--wd1', type=float, default=0.001, help='weight decay (L2 loss on parameters).')
parser.add_argument('--wd2', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
parser.add_argument('--layer', type=int, default=64, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=64, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=3000, help='Patience')
parser.add_argument('--data', default='cora', help='dateset')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--alpha', type=float, default=0.4, help='alpha_l')
parser.add_argument('--lamda', type=float, default=0.5, help='lamda.') 
parser.add_argument('--OPname', type=str, default='./Points/optest.txt', help='name of origin point data file')
parser.add_argument('--variant', action='store_true', default=True, help='GCN* model.')
parser.add_argument('--test', action='store_true', default=False, help='evaluation on test set.')
parser.add_argument('--no-cuda', action='store_true', default=False,help='Disables CUDA training.')
parser.add_argument('--percenttrain', type=float, default=0.1, help='percent of train data')
parser.add_argument('--INpaint', type=str, default='./Points/inpaint.txt', help='name of inpaint point file')
parser.add_argument('--OUTpaint', type=str, default='./Points/outpaint.txt', help='name of outpaint point file')
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

args.sceneHG=False
# dis, adj, features, labels, idx_train, idx_val, idx_test, SPidx,idx,labelweight= KNN()
xyzadj, lamdaadj, featurexyz, featurelamda, labels, idx_train, idx_val, idx_test, SPidx,idx,labelweight,numpoint = SSGCC()

cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid)

model = SSCGCN(nfeatxyz=featurexyz.shape[1],
                    nfeatlamda=featurelamda.shape[1],
                    nlayers=args.layer,
                    nhidden=args.hidden,
                    nclass=int(labels.max()) + 1,
                    dropout=args.dropout,
                    lamda = args.lamda, 
                    alpha=args.alpha,
                    npoint=numpoint)

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    model.cuda()
    # features = features.cuda()
    featurexyz = featurexyz.cuda()
    featurelamda = featurelamda.cuda()
    xyzadj = xyzadj.cuda()
    lamdaadj = lamdaadj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    idx1=idx[0].cuda()
    idx2=idx[1].cuda()
    idx3=idx[2].cuda()
    idx4=idx[3].cuda()
    idx5=idx[4].cuda()
    idx6=idx[5].cuda()
    idx7=idx[6].cuda()
    idx8=idx[7].cuda()
    if args.sceneHG:
        idx9=idx[8].cuda()

    labelweight=labelweight.cuda()
checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'
print(cudaid,checkpt_file)

optimizer = optim.Adam([
                        {'params':model.params1,'weight_decay':args.wd1},
                        {'params':model.params2,'weight_decay':args.wd2},
                        ],lr=args.lr)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs/5, eta_min=0.00001)

def train():
    model.train()
    optimizer.zero_grad()
    output = model(featurexyz, featurelamda, xyzadj, lamdaadj)



    loss_1=F.nll_loss(output[idx1], labels[idx1])
    loss_2=F.nll_loss(output[idx2], labels[idx2])
    loss_3=F.nll_loss(output[idx3], labels[idx3])
    loss_4=F.nll_loss(output[idx4], labels[idx4])
    loss_5=F.nll_loss(output[idx5], labels[idx5])
    loss_6=F.nll_loss(output[idx6], labels[idx6])
    loss_7=F.nll_loss(output[idx7], labels[idx7])
    loss_8=F.nll_loss(output[idx8], labels[idx8])
    if args.sceneHG:
        loss_9=F.nll_loss(output[idx9], labels[idx9])
        loss_train=loss_1*labelweight[0]+loss_2*labelweight[1]+loss_3*labelweight[2]+loss_4*labelweight[3]+loss_5*labelweight[4]+loss_6*labelweight[5]+loss_7*labelweight[6]+loss_8*labelweight[7]+loss_9*labelweight[8]    
    else:
        loss_train=loss_1*labelweight[0]+loss_2*labelweight[1]+loss_3*labelweight[2]+loss_4*labelweight[3]+loss_5*labelweight[4]+loss_6*labelweight[5]+loss_7*labelweight[6]+loss_8*labelweight[7]   
    
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    return loss_train.item(),acc_train.item()


def validate():
    model.eval()
    with torch.no_grad():
        output = model(featurexyz, featurelamda, xyzadj, lamdaadj)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
        acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
        return loss_val.item(),acc_val.item()

def test():
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = model(featurexyz, featurelamda, xyzadj, lamdaadj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
    return loss_test.item(),acc_test.item(),output
    
t_total = time.time()
bad_counter = 0
best = 999999999
best_epoch = 0
acc = 0





t_total = time.time()
x=[]
train_loss_list=[]
val_loss_list=[]
train_acc_list=[]
val_acc_list=[]

for epoch in range(args.epochs):
    loss_tra,acc_tra = train()
    loss_val,acc_val = validate()
    if(epoch+1)%1 == 0: 
        print('Epoch:{:04d}'.format(epoch+1),
            'train',
            'loss:{:.3f}'.format(loss_tra),
            'acc:{:.2f}'.format(acc_tra*100),
            '| val',
            'loss:{:.3f}'.format(loss_val),
            'acc:{:.2f}'.format(acc_val*100))
        


    train_loss_list.append(loss_tra)
    val_loss_list.append(loss_val)
    train_acc_list.append(acc_tra)
    val_acc_list.append(acc_val)
    x.append(epoch)
    scheduler.step()


    if loss_val < best:
        best = loss_val
        best_epoch = epoch
        acc = acc_val
        torch.save(model.state_dict(), checkpt_file)
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

# if args.test:
loss,acc,output=test()


args.Plot=False
if args.Plot:
    plt.subplot(2, 1, 1)
    plt.plot(x, train_loss_list, label='train_loss')
    plt.plot(x, val_loss_list, label='val_loss')
    plt.title("loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(x, train_acc_list, label='train_acc')
    plt.plot(x, val_acc_list, label='val_acc')
    plt.title("acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()

    plt.show()





print("Train cost: {:.4f}s".format(time.time() - t_total))
print('Load {}th epoch'.format(best_epoch))
print("Test" if args.test else "Val","acc.:{:.1f}".format(acc*100))
    

output=output.cpu().detach().numpy()
predis=np.argmax(output,axis=1)
labels=labels.cpu().detach().numpy()
CaccOA=classacc(labels,predis)

print("Classification Accuracy:",CaccOA)


args.Visualisation=True
if args.Visualisation:
    print("Broadcast Labels...")
    # output=output.cpu().detach().numpy()
    # predis=np.argmax(output,axis=1)
    xyz,outlabel,inlabel=Broadcastlabel(pointname=args.OPname, SPidx=SPidx,labels=predis)
    
    CM=confusionmatrix(outlabel,inlabel)
    print("Confusion Matrix:",np.sum(CM,axis=1))
    f1=f1_score(inlabel,outlabel,average='macro')
    p=precision_score(inlabel,outlabel,average='macro')
    r=recall_score(inlabel,outlabel,average='macro')
    print("F1 score:",f1)
    print("Precision:",p)
    print("Recall:",r)
    f1=f1_score(inlabel,outlabel,average='micro')
    p=precision_score(inlabel,outlabel,average='micro')
    r=recall_score(inlabel,outlabel,average='micro')
    print("F1 score:",f1)
    print("Precision:",p)
    print("Recall:",r)
    
    print("Confusion Matrix:",CM)
    F_score(CM)
    print("Visualisation...")
    if args.sceneHG:
        paint(xyz,outlabel,txtname='./Points/HT/SSGCC_{}.txt'.format(args.percenttrain))
    else:
        paint(xyz,outlabel,txtname='./Points/UH/SSGCC_{}.txt'.format(args.percenttrain))
    print("Visualisation Finished!")
    NCM=normalize_adj(CM)
    
    if args.sceneHG:
        np.savetxt('./confuseHT/SSGCN_{}.txt'.format(args.percenttrain),NCM,delimiter=" ",fmt="%.2f")
        ax= sns.heatmap(pd.DataFrame(NCM), annot=True,square=True, cmap="YlGnBu",
        xticklabels=["Barren", "Building", "Car","Grass","Powerline", "Road", "Ship","Tree","Water"],
        yticklabels=["Barren", "Building", "Car","Grass","Powerline", "Road", "Ship","Tree","Water"],fmt=".2f")
    else:
        np.savetxt('./confuseUH/SSGCC_{}.txt'.format(args.percenttrain),NCM,delimiter=" ",fmt="%.2f")
        ax= sns.heatmap(pd.DataFrame(NCM), annot=True,square=True, cmap="YlGnBu",
        xticklabels=["Barren", "Car", "Commercial Building","Grass","Road", "Powerline", "Residential Building","Tree"],
        yticklabels=["Barren", "Car", "Commercial Building","Grass","Road", "Powerline", "Residential Building","Tree"],fmt=".2f")
    plt.show()
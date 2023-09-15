import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from arguments import args

class MainGraphConvolution(nn.Module):

    def __init__(self, in_features, out_features):
        super(MainGraphConvolution, self).__init__() 
        self.in_features = in_features
        self.out_features = out_features
        # self.npoint = npoint
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        # self.alpha = Parameter(torch.FloatTensor(1))
        # self.adjattention = Adjattention(self.npoint)
        # self.alpha = Parameter(torch.FloatTensor(1))
        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
        # self.alpha.data.fill_(0)

    def forward(self, input, adj , Rxyz, Rlamda , t, alpha, l):
        theta = math.log(t/l+1)

        # alpha = self.alpha.data
        # alpha = torch.sigmoid(alpha)/2
        # self.alpha.data.copy_(alpha)
        # adj = self.adjattention(adj1,adj2)
        hi = torch.spmm(adj, input)
        h0 = torch.cat([Rxyz,Rlamda],1)
        support = torch.cat([hi,h0],1)
        # support =torch.cat([Rxyz,hi,Rlamda],1)
        r = (1-alpha)*hi+alpha*h0
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        return output

class SubGraphConvolution(nn.Module):
    
    def __init__(self, in_features, out_features):
        super(SubGraphConvolution, self).__init__() 
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
    
    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        return output
    
class Adjattention(nn.Module):

    def __init__(self, npoint):
        super(Adjattention, self).__init__() 
        self.npoint = npoint
        self.weight = Parameter(torch.FloatTensor(self.npoint,self.npoint))
        self.reset_parameters()
    
    def reset_parameters(self):
        self.weight.data.uniform_(0, 1)
    
    def forward(self, xyzadj, lamdaadj):
        weight = self.weight.data
        # weight = torch.clamp(weight,0,1)
        weight = torch.sigmoid(weight)
        self.weight.data.copy_(weight) 
        adj = weight*xyzadj+(1-weight)*lamdaadj
        return adj

    


class SSCGCN(nn.Module):
    def __init__(self, nfeatxyz, nfeatlamda, nlayers,nhidden, nclass, dropout, lamda, alpha,npoint):
        super(SSCGCN, self).__init__()
        self.npoint = npoint
        nfeat=nfeatxyz+nfeatlamda
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(MainGraphConvolution(4*nhidden, 2*nhidden))
        self.fcs = nn.ModuleList()
        self.fcs.append(SubGraphConvolution(nfeatxyz, nhidden))
        self.fcs.append(SubGraphConvolution(nfeatlamda, nhidden))
        self.fcs.append(nn.Linear(nfeat, 2*nhidden))
        self.fcs.append(nn.Linear(2*nhidden, nclass))
        self.adjattention = Adjattention(self.npoint)
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.t = lamda
        self.mask = Parameter(torch.FloatTensor(self.npoint,self.npoint))


    def forward(self, featurexyz,featurelamda,xyzadj,lamdaadj):
        _layers = []
        x = torch.cat([featurexyz,featurelamda],1)
        layer_inner=self.act_fn(self.fcs[2](x))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        adj=self.adjattention(xyzadj,lamdaadj)
        Rxyz= self.act_fn(self.fcs[0](featurexyz,lamdaadj))
        _layers.append(Rxyz)
        Rlamda = self.act_fn(self.fcs[1](featurelamda,xyzadj))
        _layers.append(Rlamda)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],_layers[1],self.t,self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return F.log_softmax(layer_inner, dim=1)
    



if __name__ == '__main__':
    pass







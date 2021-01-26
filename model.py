from torchtools import *
from collections import OrderedDict
import math
#import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import dgl
import torch
from mlp_layer import MLPReadout

class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, userelu=True, momentum=0.1, affine=True, track_running_stats=True):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('Conv', nn.Conv2d(in_planes, out_planes,
            kernel_size=3, stride=1, padding=1, bias=False))

        if tt.arg.normtype == 'batch':
            self.layers.add_module('Norm', nn.BatchNorm2d(out_planes, momentum=momentum, affine=affine, track_running_stats=track_running_stats))
        elif tt.arg.normtype == 'instance':
            self.layers.add_module('Norm', nn.InstanceNorm2d(out_planes))

        if userelu:
            self.layers.add_module('ReLU', nn.ReLU(inplace=True))

        self.layers.add_module(
            'MaxPool', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

    def forward(self, x):
        out = self.layers(x)
        return out

class ConvNet(nn.Module):
    def __init__(self, opt, momentum=0.1, affine=True, track_running_stats=True):
        super(ConvNet, self).__init__()
        self.in_planes  = opt['in_planes']
        self.out_planes = opt['out_planes']
        self.num_stages = opt['num_stages']
        if type(self.out_planes) == int:
            self.out_planes = [self.out_planes for i in range(self.num_stages)]
        assert(type(self.out_planes)==list and len(self.out_planes)==self.num_stages)

        num_planes = [self.in_planes,] + self.out_planes
        userelu = opt['userelu'] if ('userelu' in opt) else True

        conv_blocks = []
        for i in range(self.num_stages):
            if i == (self.num_stages-1):
                conv_blocks.append(
                    ConvBlock(num_planes[i], num_planes[i+1], userelu=userelu))
            else:
                conv_blocks.append(
                    ConvBlock(num_planes[i], num_planes[i+1]))
        self.conv_blocks = nn.Sequential(*conv_blocks)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv_blocks(x)
        out = out.view(out.size(0),-1)
        return out



# encoder for imagenet dataset
class EmbeddingImagenet(nn.Module):
    def __init__(self,
                 emb_size):
        super(EmbeddingImagenet, self).__init__()
        # set size
        self.hidden = 64
        self.last_hidden = self.hidden * 25
        self.emb_size = emb_size

        # set layers
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=3,
                                              out_channels=self.hidden,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=self.hidden,
                                              out_channels=int(self.hidden*1.5),
                                              kernel_size=3,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=int(self.hidden*1.5)),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=int(self.hidden*1.5),
                                              out_channels=self.hidden*2,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 2),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.4))
        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels=self.hidden*2,
                                              out_channels=self.hidden*4,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 4),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.5))
        self.layer_last = nn.Sequential(nn.Linear(in_features=self.last_hidden * 4,
                                              out_features=self.emb_size, bias=True),
                                        nn.BatchNorm1d(self.emb_size))

    def forward(self, input_data):
        output_data = self.conv_4(self.conv_3(self.conv_2(self.conv_1(input_data))))
        return self.layer_last(output_data.view(output_data.size(0), -1))




class NodeUpdateNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 num_features,
                 L,
                 dropout=0.0,):
        super(NodeUpdateNetwork, self).__init__()
        # set size
        self.in_features = in_features
        #self.num_features_list = [num_features * r for r in ratio]
        self.dropout = dropout


        self.in_channels = in_features
        self.out_channels = in_features
        self.dropout = dropout
        self.batch_norm = True
        self.residual = True
        self.L=L
        if self.L==0:
           self.embedding_e = nn.Linear(2, 96)
        self.A = nn.Linear(in_features, in_features, bias=True)
        self.B = nn.Linear(in_features, in_features, bias=True)
        self.C = nn.Linear(96, 96, bias=True)
        self.D = nn.Linear(in_features, in_features, bias=True)
        self.E = nn.Linear(in_features, in_features, bias=True)
        self.bn_node_h = nn.BatchNorm1d(in_features)
        self.bn_node_e = nn.BatchNorm1d(in_features)
        self.embedding_out =MLPReadout(96, 2)
        #self.embedding_out = nn.Linear(128,2)
        self.e_GRU1 = torch.nn.GRUCell(96,96)
        self.e_GRU2 = torch.nn.GRUCell(96,96)

    def message_func(self, edges):
        #print(edges.src['Bh'].size(),edges.data['Ce'].size(),edges.src['Dh'].size(),edges.dst['Eh'].size())
        Bh_j = edges.src['Bh']    
        
        #e_ij = e_GRU2(edges.dst['Eh'],e_GRU1(edges.src['Dh'],edges.data['e']))
        #e_ij = edges.data['Ce'] +  edges.src['Dh'] + edges.dst['Eh'] # e_ij = Ce_ij + Dhi + Ehj
        e_ij = edges.data['e']
        return {'Bh_j' : Bh_j, 'e_ij' : e_ij}

    def reduce_func(self, nodes):
        Ah_i = nodes.data['Ah']
        Bh_j = nodes.mailbox['Bh_j']
        e = nodes.mailbox['e_ij'] 
        sigma_ij = torch.sigmoid(e) # sigma_ij = sigmoid(e_ij)
        #h = Ah_i + torch.mean( sigma_ij * Bh_j, dim=1 ) # hi = Ahi + mean_j alpha_ij * Bhj 
        h = Ah_i + torch.sum( sigma_ij * Bh_j, dim=1 ) / ( torch.sum( sigma_ij, dim=1 ) + 1e-6 )  # hi = Ahi + sum_j eta_ij/sum_j' eta_ij' * Bhj <= dense attention       
        return {'h' : h}
  
    def edge_udf(self, edges):
        # edges.src['h'] is a tensor of shape (E, 1),
        # where E is the number of edges in the batch.
        return {'src': edges.src['h'],'dst': edges.dst['h']}

    
    def forward(self, g, node_feat, edge_feat):
        if self.L==0:
           e = self.embedding_e(edge_feat)
        else:
           e=edge_feat
        #h_in = node_feat.to("cuda:0") # for residual connection
        #e_in = edge_f.to("cuda:0") # for residual connection
        h_in = node_feat
        e_in = e
        #print(g.edges(),edge_f.size(),node_feat.size())
        g.ndata['h']  = node_feat 
        g.ndata['Ah'] = self.A(node_feat) 
        g.ndata['Bh'] = self.B(node_feat) 
        #g.ndata['Dh'] = self.D(node_feat)
        #g.ndata['Eh'] = self.E(node_feat) 
        g.edata['e']  = e
        #g.edata['Ce'] = self.C(e) 
        g.update_all(self.message_func,self.reduce_func) 
        h = g.ndata['h'] # result of graph convolution
        #e = g.edata['e'] # result of graph convolution
        g.apply_edges(self.edge_udf)
        #print(g.edata['src'].size(),g.edata['dst'].size())
        e = self.e_GRU2(g.edata['dst'],self.e_GRU1(g.edata['src'],g.edata['e']))
        g.edata['e']  = e
        

        h = self.bn_node_h(h) # batch normalization  
        e = self.bn_node_e(e) # batch normalization  
        
        h = F.relu(h) # non-linear activation
        e = F.relu(e) # non-linear activation
        
        #print(e.size(),e_in.size())
        #if self.residual:
        h = h_in + h # residual connection
        e = e_in + e # residual connection
        
        #h = F.dropout(h, self.dropout)
        #e = F.dropout(e, self.dropout)
        if self.L==2:
           e = self.embedding_out(e)
        return g,h,e 




class GraphNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 node_features,
                 edge_features,
                 num_layers,
                 dropout=0.0):
        super(GraphNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.node_features = node_features
        self.edge_features = edge_features
        #self.num_layers = num_layers
        #self.dropout = dropout 


        self.out0=NodeUpdateNetwork(in_features=self.in_features, num_features=self.node_features,L=0)
        self.out1=NodeUpdateNetwork(in_features=self.in_features, num_features=self.node_features,L=1)
        self.out2=NodeUpdateNetwork(in_features=self.in_features, num_features=self.node_features,L=2)

    # forward
    def forward(self, g, node_feat, edge_feat):

        #edge_feat_list = []
        g0,h0,e0 = self.out0(g, node_feat, edge_feat)
        g1,h1,e1 = self.out1(g0, h0, e0)
        g2,h2,e2 = self.out2(g1, h1, e1)
        edge_feat_list = [e0,e1,e2]

        return edge_feat_list


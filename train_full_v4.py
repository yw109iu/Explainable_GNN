
import pickle
import sys
import timeit
import random

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx
import copy

from sklearn.metrics import roc_auc_score, precision_score, recall_score, cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from datetime import datetime 


class myLinear_1(nn.Module):
    def __init__(self, A, y,Adjs): #bias=True
        super().__init__()
        self.A = A
        self.y = y
        #self.bias = bias
        self.weight = torch.nn.Parameter(A)
        #self.bias = torch.nn.Parameter(torch.randn(A.shape[0],y))
        self.weights = []
        for i in Adjs:
            self.weights.append(torch.nn.Parameter(i))
            
    def pad(self, matrices, pad_value):
        """Pad adjacency matrices for batch processing."""
        sizes = [d.shape[0] for d in matrices]
        D = sum(sizes)
        pad_matrices = pad_value + np.zeros((D, D))
        m = 0
        for i, d in enumerate(matrices):
            s_i = sizes[i]
            #Fill in the pad_matrics, block by block
            pad_matrices[m:m+s_i, m:m+s_i] = d
            m += s_i
        return torch.FloatTensor(pad_matrices).to(device)        
    
    def forward(self, input,batch_idrange):
        current_weights = []
        for ele in batch_idrange:
            current_weights.append(self.weights[ele])
        weights = self.pad(current_weights, 0)
        output =  torch.matmul(weights,input ) #+ self.bias
        return output 
    
class myLinear(nn.Module):
    def __init__(self, A): #bias=True
        super().__init__()
        self.A = A
        self.weight = torch.nn.Parameter(A) 
        #self.bias = torch.nn.Parameter(torch.randn(A.shape[0],y))      
    
    def forward(self, input_):
        output =  torch.matmul(self.weight.to(device),input_) #+ self.bias
        return output #n*n x n*d = n*d
    
class myLinear_last(nn.Module):
    def __init__(self): #bias=True
        super().__init__()
        #self.A = A
        #self.y = y
        #self.bias = bias
        #self.weight = torch.nn.Parameter(A)
        w = torch.empty(7686) 
        #Lyn_noTree 7725 #Lck_noTree 4957 #Src_noTree 7686
        #7725 LYN 211 #7737 lyn with 5 case studies #7629/7686 for scr no case study #4903/4957 for lck no case study
        w_ = nn.init.ones_(w)
        #print(w_)
        self.w_1 = torch.diag(w_).to(device)
        self.w_1 = torch.nn.Parameter(self.w_1)
        #self.bias = torch.nn.Parameter(torch.randn(A.shape[0],y))      
    
    def forward(self, input):
        #self.weights = torch.nn.Parameter(torch.zeros(adjacencies.shape)).to(device)
        output =  torch.matmul(self.w_1.to(device), input)#torch.matmul(self.weight,input ) #+ self.bias
        #print('vvvvvvvvv~',output.shape)
        return output #n*n x n*d = n*d

class GraphNeuralNetwork(nn.Module):
    def __init__(self,A,init_emb): #input where A is the grand Adj matrix
        super(GraphNeuralNetwork, self).__init__()
        self.embed_fingerprint = init_emb#nn.Embedding(n_fingerprint, dim)
        #self.embed_fingerprint.weight.requires_grad = False
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim)
                                            for _ in range(hidden_layer)])
        self.W_output_noB = nn.ModuleList([nn.Linear(dim, dim)
                                       for _ in range(output_layer)])
        self.W_fingerprint_weigts = myLinear_last
        self.embed_fingerprint.weight.requires_grad = True
        self.graph_structurelayer = myLinear(adjacencies)  
        ############################################################
        self.W_property_noB = nn.Linear(dim, 2)

    def pad(self, matrices, pad_value):
        """Pad adjacency matrices for batch processing."""
        sizes = [d.shape[0] for d in matrices]
        D = sum(sizes)
        pad_matrices = pad_value + np.zeros((D, D))
        m = 0
        for i, d in enumerate(matrices):
            s_i = sizes[i]
            #Yang: fill in the pad_matrics, block by block
            pad_matrices[m:m+s_i, m:m+s_i] = d
            m += s_i
        return torch.FloatTensor(pad_matrices).to(device)

    def sum_axis(self, xs, axis):
        y = list(map(lambda x: torch.sum(x, 0), torch.split(xs, axis)))
        return torch.stack(y)

    def mean_axis(self, xs, axis):
        y = list(map(lambda x: torch.mean(x, 0), torch.split(xs, axis)))
        return torch.stack(y)

    def gnn(self, xs, A, M, i):
        hs = torch.relu(self.W_fingerprint[i](xs))
        #print('~~~~~~Check hs:~',hs.shape,hs[0])
        #print('~~~~~~Check xs:~',xs.shape)
        #sys.exit() 
        if update == 'sum':
            return xs + self.graph_structurelayer.forward(hs)
        if update == 'mean':
            return xs + self.graph_structurelayer.forward(hs)/(M-1)

    def forward(self, inputs):
        
        Smiles, fingerprints, adjacencies = inputs #docking_scores
        axis = list(map(lambda x: len(x), fingerprints)) #axis equal to the number of mols
        #print('Check what is axis:~',len(axis),axis)
        
        M = np.concatenate([np.repeat(len(f), len(f)) for f in fingerprints])
        #print('~~~CHeck M:',M.shape)
        M = torch.unsqueeze(torch.FloatTensor(M), 1)
        #print('~~~CHeck M:',M.shape)
        #print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~CHeck forward fingerprints:~',fingerprints[9].shape)
        fingerprints = torch.cat(fingerprints)
        #print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~CHeck forward fingerprints:~',fingerprints)
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        #print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Compare to orginal Emb weight vec:~',fingerprints.shape)
        #print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~CHeck forward fingerprints vectors:~',fingerprint_vectors.shape,fingerprint_vectors[0])
        #sys.exit()
        #print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Check adjacencies:~',len(adjacencies),adjacencies[2].shape)
        adjacencies = self.pad(adjacencies, 0)
        #print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Check adjacencies:~',len(adjacencies),adjacencies.shape)
        #sys.exit()
        for i in range(hidden_layer):
            fingerprint_vectors = self.gnn(fingerprint_vectors,
                                           adjacencies, M, i)
            #torch.unsqueeze(torch.mean(xs, 0), 0)
            #print('rebuild_fingerprint:~',fingerprint_vectors.shape)
            #sys.exit()
        if output == 'sum':
            molecular_vectors = self.sum_axis(fingerprint_vectors, axis)
            #print('Sum molecular_vec:~',molecular_vectors.shape)
        if output == 'mean':
            
            #W_fingerprint_weights !!!!!!
            zeros_out_indes = np.where(~self.graph_structurelayer.weight.cpu().any(axis=1))[0]
            zeros_ = np.zeros((self.graph_structurelayer.weight.cpu()[zeros_out_indes].shape[0],self.graph_structurelayer.weight.cpu().shape[1]))
            #print(zeros_out_indes,zeros_out_indes.shape,len(zeros_out_indes))
            if len(zeros_out_indes) !=0:
                #print(self.W_fingerprint_weigts.w_1[zeros_out_indes],zeros_out_indes,zeros_)
                zeros_ = torch.from_numpy(zeros_).to(device)
                with torch.no_grad():
                    zeros_ = zeros_.float()
                    self.W_fingerprint_weigts.w_1[zeros_out_indes] = zeros_
                    #print(self.W_fingerprint_weigts.w_1.requires_grad)
                    #sys.exit()
            fingerprint_vectors = self.W_fingerprint_weigts.forward(fingerprint_vectors)
            
            #print('Check finger_vectors shape:~',fingerprint_vectors.shape)
            molecular_vectors = self.mean_axis(fingerprint_vectors, axis)
            #print('Mean molecular_vec::~',molecular_vectors.shape)

        #print(molecular_vectors)
        """getting docking scores and concatenate them with molecular vectors"""
        #docking_scores = torch.from_numpy(np.asarray(docking_scores)).to(device)
        #y_cat = torch.cat((docking_scores, molecular_vectors), 1)
        #print(y_cat.shape,docking_scores.shape)
        y_cat = molecular_vectors
        
        for j in range(output_layer):
            y_cat = torch.relu(self.W_output_noB[j](y_cat))

        predicted_properties = self.W_property_noB(y_cat)

        return Smiles, predicted_properties,molecular_vectors



    def __call__(self, data_batch,train=True):

        inputs = data_batch[:-1]
        correct_properties = torch.cat(data_batch[-1])
        Smiles, predicted_properties,molecular_vectors = self.forward(inputs)

        if train:
            #loss = F.cross_entropy(predicted_properties, correct_properties)
            loss = F.cross_entropy(predicted_properties, correct_properties)
            return loss
        else:
            ts = correct_properties.to('cpu').data.numpy()
            ys = F.softmax(predicted_properties, 1).to('cpu').data.numpy()
            correct_labels = ts
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            return correct_labels, predicted_labels, predicted_scores,molecular_vectors,ts
        
##From Non-Convex GTF, "Vector-Valued Graph Trend Filtering"
class GroupProximalOperator(object):
    def __init__(self, param=None):
        self.param = param

    def threshold(self, V, gamma):
        """
        returns the answer to prox_{gamma, f}(v) = argmin_x (f(x) + 1/2gamma ||x-v||_2^2)
        """
        pass

    def sign(self, x):
        result = x.clone()#.copy()
        result[x > 0] = 1.0
        result[x < 0] = -1.0
        return result

    def soft_threshold(self, V, gamma):
        v_norms = np.linalg.norm(V, axis=1) # row-wise norm
        shrink = np.zeros(v_norms.shape)
        print('Check v_norms:~',shrink)
        #sys.exit()
        shrink[v_norms>0] = np.maximum(1 - gamma/v_norms[v_norms>0], np.zeros(v_norms[v_norms>0].shape))
        return np.dot(np.diag(shrink), V)
    
    def soft_threshold_Yang(self,x,gamma):
        re_ = self.sign(x) * torch.maximum(torch.abs(x)-gamma,torch.zeros(x.shape).to(device)) 
        return re_ #   #tf.maximum(tf.abs(x) - threshold, 0.)
    
    def soft_threshold_Yang_v2(self,x,gamma):
        gamma = torch.Tensor([gamma]).to(device)
        gamma = torch.transpose(gamma,0,1)
        #print('Inside soft:~',gamma,gamma.shape)
        #print('Inside soft:~',x.shape,torch.abs(x).shape,torch.abs(x)-gamma)
        re_ = self.sign(x) * torch.maximum(torch.abs(x)-gamma,torch.zeros(x.shape).to(device)) 
        return re_ #   #tf.maximum(tf.abs(x) - threshold, 0.)
    
##From Non-Convex GTF, "Vector-Valued Graph Trend Filtering"
class proximal_l1(GroupProximalOperator):
    def threshold(self, X, gamma):
        """
        returns the answer to prox_{gamma}(v) = argmin_x (||x||_1 + 1/2gamma ||x-v||_2^2)
        This is just soft thresholding with parameter gamma.
        """
        return self.soft_threshold_Yang(X, gamma)
    
##From Non-Convex GTF, "Vector-Valued Graph Trend Filtering"
class Penalty(object):
    def __init__(self, param=None):
        self.param = param
        self.maxValues = {}

    def calculate(self, x, gamma):
        """
        calculates the magnitude of the penalty term
        """
        pass

##From Non-Convex GTF, "Vector-Valued Graph Trend Filtering"
def L1Penalty_calculate(x, gamma):
        """
        returns gamma*|x|
        """
        return gamma*torch.norm(x, p=1)

#Yang, version-2:     
def L1Penalty_calculate_v2(x, gamma):
        """
        returns gamma*|x|
        """
        res_ = 0
        for freq_,ele in zip(gamma,x):
            res_ = res_ +freq_*torch.abs(ele)
        
        
        return res_[0]

def L2Power2Penalty_calculate(x, gamma):
        """
        returns gamma*|x|
        """
        return gamma*(torch.norm(x, p=2)*torch.norm(x, p=2))
    
def build_edgegraph_incidenceMt(Adj):
    G = nx.from_numpy_matrix(Adj)
    L = nx.line_graph(G)
    incidence_matrix = (nx.incidence_matrix(L,oriented=True)).toarray().transpose()
    return torch.from_numpy(incidence_matrix).double().to(device)

def Adj_update(beta_vec,G):
    edges = beta_vec
    edges_indexs = torch.nonzero(torch.triu(G))
    aux = 0 
    G1 = G.clone()
    for row_id_col_id in edges_indexs:
        if edges[aux] != 0:
            tmp = 1
        elif edges[aux] == 0:
            tmp = 0
        G1[row_id_col_id[0],row_id_col_id[1]] = tmp
        G1[row_id_col_id[1],row_id_col_id[0]] = tmp
        aux= aux+1
    
    return G1


def Adj_update_v1(beta_vec,G):
    edges = beta_vec
    edges_indexs = np.nonzero(G)
    aux = 0 
    for ele in edges_indexs:
        #edges.append(link_graph_nodes[ele])
        G[ele[0]][ele[1]] = edges[aux]
        G[ele[1]][ele[0]] = edges[aux]
        aux= aux+1
    print('Check aux:~',aux,torch.count_nonzero(G))
    return G


class Trainer(object):
    def __init__(self, model,beta_intt_indexs,freqs_for_mols):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)
        self.reg_lambda3 = 0.01
        self.beta_intt_indexs = beta_intt_indexs
        self.freqs_for_mols = freqs_for_mols
        
    def proximal_glasso(self,yvec, c):
        #Uk = Wembk - 1/ck derivitivesLossWemb #Wemb,Wgnn
        ynorm = torch.linalg.norm(yvec, ord=2) #fro
        if ynorm > c/2.:
            xvec = (yvec/ynorm)*(ynorm-c/2.)
        else:
            xvec = torch.zeros_like(yvec)
        return xvec
    def proximal_l2(self,yvec, c):
        return (1./(1.+c))*yvec
    
    def proximal_l0(self,yvec, c):
        yvec_abs =  torch.abs(yvec)
        csqrt = torch.sqrt(2*c)
        xvec = (yvec_abs>=csqrt)*yvec
        return xvec
    

    def train(self, dataset,lr_,lip):
        #np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        l1_loss = 0
        edgel1_loss = 0
        weightdecay_loss = 0
        #reg_l0=0.0001
        reg_glasso= 1 #1 #10 #0.1
        reg_glasso_0= 9 #9  #9.7
        #78#100#99.999#9.999  #~9.7~  #99.99999999999999999999 #0.1 #17 ###47 for Lyn214, 47 for Lyn1225 ###60, for lck  ###55, for SCR
        reg_decay=1e-3
        reg_l0 = 45 #45
        lr_ = 0.01  #0.01
        #lr_= 0.02  #0.01
        lip= 0.01  #0.005
        shape_ = check_network_fussedlasso(model,dim)
        print('Before all prune shape_',shape_)
        #sys.exit()
        #block_indexs
        def grad_hook_masking(grad, mask):
            grad = grad.mul_(mask)
            del mask
            return grad
        for i in range(0, N, batch):
            #print('New batch~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            data_batch = list(zip(*dataset[i:i+batch]))
            block_indexs[i:i+batch]
            #frozen the GNN A parameter when pretrain
            #for name, param in self.model.named_parameters():
            #    if 'graph_structurelayer' in name:
            #        param.requires_grad = False
            self.optimizer.zero_grad()
            handle_list_0 = []
            
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if 'graph_structurelayer' in name:
                        mask = torch.where(param.data.detach()!=0, torch.ones_like(param.data.detach()), torch.zeros_like(param.data.detach()))
                        #sys.exit()
                        handle = param.register_hook(lambda grad, mask=mask: grad_hook_masking(grad, mask))
                        handle_list_0.append(handle)
                        
                        betarize_param_tmp = torch.unsqueeze(param.data[self.beta_intt_indexs], dim=0).t().double()
                        aux_ = 0
                        new_lamdas = []
                        for freq_, ele in zip(freqs_for_mols,betarize_param_tmp):
                            new_lambda = reg_glasso_0 - self.reg_lambda3*freq_
                            new_lamdas.append(new_lambda)
                            aux_ = aux_ +1                  
                        
                    else:
                        #other k2 weight decay
                        mask = torch.where(param.data.detach()!=0, torch.ones_like(param.data.detach()), torch.zeros_like(param.data.detach()))
                        handle = param.register_hook(lambda grad, mask=mask: grad_hook_masking(grad, mask))
                        #handle = param.register_hook(lambda grad: grad.mul_(torch.where(param.data.detach()!=0, torch.ones_like(param.data.detach()), torch.zeros_like(param.data.detach()))))
                        handle_list_0.append(handle)
            
            loss = self.model(data_batch)
            #loss_total += loss.to('cpu').data.numpy()
            loss.backward()
            #loss_total.backward()
            #l1_loss = L1Penalty_calculate(x, gamma)
            #edgel1_loss = L1Penalty_calculate
            #weightdecay_loss = torch.norm(a)
            
            #handle.remove() 
            #print(len(torch.nonzero(torch.triu(model.graph_structurelayer.weight.grad)+torch.t(torch.tril(model.graph_structurelayer.weight.grad))-torch.diag(torch.diag(model.graph_structurelayer.weight.grad)))))
            
            #tmp = torch.triu(model.graph_structurelayer.weight.grad)+ torch.t(torch.tril(model.graph_structurelayer.weight.grad))- torch.diag(torch.diag(model.graph_structurelayer.weight.grad)) 
            #model.graph_structurelayer.weight.grad.data = tmp+torch.t(torch.triu(model.graph_structurelayer.weight.grad))-torch.diag(torch.diag(model.graph_structurelayer.weight.grad))
            
            #print('check:~',model.graph_structurelayer.weight.grad)
            #model.graph_structurelayer.weight.grad = None
            #print('heck:~',model.graph_structurelayer.weight.grad)
            ####################################################################Pre-train version
            #self.optimizer.step()
            #loss_total += loss.to('cpu').data.numpy()
            ###################################################################
            #Group lasso version
            '''
            for name, param in self.model.named_parameters():
            #print('Check model paras:',name)
                if "embed" in name:
                    for i in range(len(param)):
                        #print('Check model paras:',param.data[i].shape)
                        #sys.exit()
                        temp_input = param.data[i]
                        temp_input_grad = param.grad.data[i]
                        param_tmp = temp_input - lip*temp_input_grad
                        param_tmpupdate = self.proximal_glasso(param_tmp, reg_glasso*lip)
                        param.data[i] = param_tmpupdate
                elif 'W_fingerprint' in name:
                    param_tmp = param.data - lr_*param.grad.data
                    #param.data = param_tmp
                    param.data = self.proximal_l2(param_tmp, 2*reg_decay*lr)
                    #print('Check model paras:',name)
                elif 'W_output' in name:
                    param_tmp = param.data - lr_*param.grad.data
                    #param.data = param_tmp
                    param.data = self.proximal_l2(param_tmp, 2*reg_decay*lr)
                    #print('Check model paras1:',name)
                elif 'W_property' in name:
                    param_tmp = param.data - lr_*param.grad.data
                    #param.data = param_tmp
                    param.data = self.proximal_l2(param_tmp, 2*reg_decay*lr)
                    #print('Check model paras:',name,param)
                    #print('JJJ~',param.grad)
                    #print(':~',model.W_fingerprint[0].weight.grad)
                loss_total += loss.to('cpu').data.numpy()
            '''
            ####################################
            ###Fussed lasso version
            prox = proximal_l1()
            for name, param in self.model.named_parameters():
                if 'graph_structurelayer' in name:
                    tmp = torch.triu(param.grad.data)+torch.t(torch.tril(param.grad.data))-torch.diag(torch.diag(param.grad.data))
                    param.grad.data = tmp+torch.t(torch.triu(tmp))-torch.diag(torch.diag(tmp))
                    #####################
                    param_tmp = param.data - lip*(param.grad.data)#*adjacencies
                    neg_param_tmp = lip*(param.grad.data) - param.data 

                    betarize_param_tmp = torch.unsqueeze(param_tmp[self.beta_intt_indexs], dim=0).t().double()
                    freqs_for_mols
                    aux_ = 0

                    betarize_param_tmp_update = prox.soft_threshold_Yang(torch.matmul(D_incidence,betarize_param_tmp),reg_glasso*lip ) #reg_glasso*lip
                    betarize_param_tmp_update = torch.matmul(I_DTD_inv, rho*torch.matmul(D_incidence.t(),(betarize_param_tmp_update)) + betarize_param_tmp)
                    ###########################################################
                    Twice_update = prox.soft_threshold_Yang_v2(betarize_param_tmp_update,[i * lip for i in new_lamdas])
                    p_ = param.data.clone()
                    param.data = Adj_update(Twice_update,adjacencies) #adjacencies.to(device)
                    l1_loss = L1Penalty_calculate_v2(Twice_update, new_lamdas)
                    edgel1_loss = L1Penalty_calculate(torch.matmul(D_incidence,Twice_update),reg_glasso)
                    loss_total += l1_loss+edgel1_loss
            
                else:
                    param_tmp = param.data - lr_*param.grad.data
                    param.data = self.proximal_l2(param_tmp, 2*reg_decay*lr_)
                    weightdecay_loss = torch.norm(param.data)*reg_decay
                    loss_total += weightdecay_loss
                loss_total = loss_total
            
                #sys.exit()
            loss = self.model(data_batch)   
            loss_total = loss + loss_total
        
        ########################################
        #sys.exit()
        #shape_ = check_network_fussedlasso(model,dim)
        #print('After all prune shape_',shape_)

        for handle in handle_list_0:
            handle.remove()
        del loss
        torch.cuda.empty_cache()
        return loss_total.to('cpu').data.numpy() ,l1_loss.to('cpu').data.numpy()  #loss_total,l1_loss #.to('cpu').data.numpy()
    
    def retrain(self,dataset,lr_,reg_decay,CUDA_ID=0):
        #print("check network before train:")
        #check_network(model, dGc, root)
        def grad_hook_masking(grad, mask):
            grad = grad.mul_(mask)
            del mask
            return grad
            
        shape_ = check_network_fussedlasso(model)
        print('Before mask, Check shape_',shape_[0])

        N = len(dataset)
        loss_total = 0
        #shape_= check_network_fussedlasso(model)
        #print("0-check network before all retrain:",shape_[0])
        for i in range(0, N, batch):
            print('Start Retrain~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            data_batch = list(zip(*dataset[i:i+batch]))
            #cuda_labels = torch.autograd.Variable(labels.cuda(CUDA_ID))
            #shape_= check_network_fussedlasso(model)
            #print("1-check network before all retrain:",shape_)
            # Forward + Backward + Optimize
            self.optimizer.zero_grad()
            handle_list = list()
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if "graph_structurelayer" in name:
                        #g-lasson/fussed lasso
                        mask = torch.where(param.data.detach()!=0, torch.ones_like(param.data.detach()), torch.zeros_like(param.data.detach()))
                        print('###############################################',torch.count_nonzero(mask))
                        handle = param.register_hook(lambda grad, mask=mask: grad_hook_masking(grad, mask))
                        #handle = param.register_hook(lambda grad: grad.mul_(torch.where(param.data.detach()!=0, torch.ones_like(param.data.detach()), torch.zeros_like(param.data.detach()))))
                        handle_list.append(handle)
                    elif 'W_fingerprint_weigts' in name:
                        mask = torch.where(param.data.detach()!=0, torch.ones_like(param.data.detach()), torch.zeros_like(param.data.detach()))
                        print('###############################################',torch.count_nonzero(mask))
                        handle = param.register_hook(lambda grad, mask=mask: grad_hook_masking(grad, mask))
                        #handle = param.register_hook(lambda grad: grad.mul_(torch.where(param.data.detach()!=0, torch.ones_like(param.data.detach()), torch.zeros_like(param.data.detach()))))
                        handle_list.append(handle)
                        #lambda3_term = L2Power2Penalty_calculate(betarize_param_tmp, self.reg_lambda3)
                    #else:
                        #other k2 weight decay
                    #    mask = torch.where(param.data.detach()!=0, torch.ones_like(param.data.detach()), torch.zeros_like(param.data.detach()))
                    #    handle = param.register_hook(lambda grad, mask=mask: grad_hook_masking(grad, mask))
                        #handle = param.register_hook(lambda grad: grad.mul_(torch.where(param.data.detach()!=0, torch.ones_like(param.data.detach()), torch.zeros_like(param.data.detach()))))
                    #    handle_list.append(handle)
            loss = self.model(data_batch)
            loss_total += loss.to('cpu').data.numpy()
            loss.backward()
            '''
            for name, param in self.model.named_parameters():
                if 'embed_fingerprint' in name:
                    pass
                else:
                    param_tmp = param.data - lr_*param.grad.data
                    param.data = self.proximal_l2(param_tmp, 2*reg_decay*lr_)
                    weightdecay_loss = torch.norm(param.data)*reg_decay
                    loss_total += weightdecay_loss
            '''
            #loss = self.model(data_batch)
            #shape_= check_network_fussedlasso(model)
            #print("2-check network before all retrain:",shape_)
            #loss.backward()
            #shape_= check_network(model)
            #print("3-check network before all retrain:",shape_)
            #print("@check network before step:")
            #shape_ = shape_= check_network_fussedlasso(model)
            #print("3 check network before retrain:",shape_)
            self.optimizer.step()
            #shape_ = shape_= check_network_fussedlasso(model)
            #print("4 check network before retrain:",shape_)
            loss_total += loss.to('cpu').data.numpy()

        del loss
        for handle in handle_list:
            handle.remove()
        torch.cuda.empty_cache()
        return loss_total

class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):

        N = len(dataset)
        Correct_labels, Predicted_labels, Predicted_scores = [], [], []

        for i in range(0, N, batch):
            data_batch = list(zip(*dataset[i:i+batch]))

            (correct_labels, predicted_labels,
             predicted_scores,molecular_vectors,ts) = self.model(data_batch,train=False)

            Correct_labels.append(correct_labels)
            Predicted_labels.append(predicted_labels)
            Predicted_scores.append(predicted_scores)

        correct_labels = np.concatenate(Correct_labels)
        predicted_labels = np.concatenate(Predicted_labels)
        predicted_scores = np.concatenate(Predicted_scores)

        AUC = roc_auc_score(correct_labels, predicted_labels)
        precision = precision_score(correct_labels, predicted_labels)
        recall = recall_score(correct_labels, predicted_labels)
        acc = accuracy_score(correct_labels, predicted_labels)
        kappa = cohen_kappa_score(correct_labels, predicted_labels)
        
        print('Acc:~',acc)
        #sys.exit()
        return AUC, precision, recall, kappa,acc, molecular_vectors,ts

    def result_AUC(self, epoch, time, loss_train, l1_loss, AUC_valid,
                   precision_valid, recall_valid, F1_score_valid,  kappa_train, acc,file_result, shape_,shape_ww,lr_,lip):
        with open(file_result, 'a') as f:
            results_list = [AUC_valid,precision_valid,recall_valid,F1_score_valid,kappa_train,acc]
            if any(isinstance(ele, str) for ele in results_list):
                result = map(str, [epoch, time, loss_train, l1_loss,lr_, lip, AUC_valid,
                               precision_valid, recall_valid, F1_score_valid, 
                               kappa_train,acc, shape_,shape_ww])
            else:
                result = map(str, [epoch, time, loss_train, l1_loss,lr_, lip, str.format("{0:.5f}",AUC_valid),
                              str.format("{0:.5f}",precision_valid), str.format("{0:.5f}",recall_valid), str.format("{0:.5f}",F1_score_valid), 
                             str.format("{0:.5f}",kappa_train), str.format("{0:.5f}",acc), shape_,shape_ww])    
            f.write('\t'.join(result) + '\n')
    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

def load_tensor(filename, dtype, allow_pickle=True):
    return [dtype(d).to(device) for d in np.load(filename + '.npy', allow_pickle=True)]

def load_numpy(filename):
    return np.load(filename + '.npy', allow_pickle=True)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

def f1_score(precision, recall):
    if precision==0 and recall == 0:
        F1_score = 'undefined'
    else:
        F1_score = 2 * ((precision * recall) / (precision + recall))
    return F1_score

#after each batch, check the model-statistics
def check_network_grouplasso(model,dim=10):
    counter_ = 0
    for name, param in model.named_parameters():
        if "embed" in name:
            for i in range(len(param)):
                Nonzero = len(torch.nonzero(param.data[i], as_tuple =False))
                if Nonzero == dim:
                    counter_ = counter_+1
            F_norm = torch.norm(param.data)   
    return counter_,F_norm


def check_network_fussedlasso(model,dim=10):
    for name, param in model.named_parameters():
        if "graph_structurelayer" in name:
            Nonzero_ = torch.count_nonzero(param.data)
            F_norm = torch.norm(param.data)   
    return Nonzero_,F_norm #,Nonzero_ww


if __name__ == "__main__":
    start_time_ = datetime.now()
    """Hyperparameters."""
    #(DATASET, radius, update, output, dim, hidden_layer, output_layer, batch,
    #lr, lr_decay, decay_interval, weight_decay, iteration,
    #setting) = sys.argv[1:]
     
    DATASET='PDL1' #Not important, you can exchange to your prefered name, it is only for saved files' name                                                                                                                                         

    radius=0 #Fix to 0                                                                                                                                               

    update='sum'                                                                                                                                          
                                                                                                                                           
    output='mean'
    
    dim=10
    hidden_layer=1
    output_layer=2
    
    
    #############################
    batch= 553  
    #Lyn 553
    #Lck 239
    #Src 499
    ##############################
    
    lr_ = 0.01 
    lip = 0.01
    lr=0.01 
    lr_decay=0.001 
    decay_interval=999
    weight_decay= 1e-5
    
    ############
    iteration_prune = 20#10#20 #10
    iteration_retrain = 3 #20 #10 
    iteration_whole = 20#100 #300

    setting= DATASET+'--radius:'+str(radius)+'--update:'+str(update)+'--output:'+str(output)+'--dim:'+str(dim)+'--hidden_layer:'+str(hidden_layer)+'--output_layer:'+str(output_layer)+'--batch:'+str(batch)+'--lr:'+str(lr)+'--lr_decay:'+str(lr_decay)+'--decay_interval:'+str(decay_interval)+'--weight_decay:'+str(weight_decay)+'--iteration_prune:'+str(iteration_prune)+'--iteration_retrain:'+str(iteration_retrain)+'--iteration_whole:'+str(iteration_whole)     

    (dim, hidden_layer, output_layer, batch, decay_interval,
     iteration_prune,iteration_retrain,iteration_whole) = map(int, [dim, hidden_layer, output_layer, batch,
                            decay_interval, iteration_prune,iteration_retrain,iteration_whole])
    lr, lr_decay, weight_decay = map(float, [lr, lr_decay, weight_decay])

    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')
        
    """Load preprocessed data."""
    dir_input = ('Src_notree/')
    with open(dir_input + 'SRC_smiles_casestudy.txt') as f:
        Smiles = f.read().strip().split()
    print('___________________________________________________________________________________________')
    Molecules = load_tensor(dir_input + 'JTMolecules_src_casestudy', torch.LongTensor)
    print('Double Check what is Molecules:~',len(Molecules),Molecules)
    
    adjacencies = load_numpy(dir_input + 'JTadjacencies_src_casestudy')
    print('Yang check adjacencies:~',len(adjacencies))

    #docking_scores = load_numpy(dir_input + 'docking_scores')
    
    correct_properties = load_tensor(dir_input + 'properties_src_casestudy',
                                     torch.LongTensor)
    correct_properties = correct_properties #[:-1] #For lyn we didn't include the last data entry.
    #############
    output_list = list(map(int, correct_properties))
    print('WWWWWWWWWWWW',output_list)
    print(len(correct_properties))
    #sys.exit()
    
    ############
    with open(dir_input + 'JTfingerprint_dict_src_casestudy.pickle', 'rb') as f:
        fingerprint_dict = pickle.load(f)
    fingerprint_dict = load_pickle(dir_input + 'JTfingerprint_dict_src_casestudy.pickle')
    n_fingerprint = len(fingerprint_dict)
    print('finger print:~',len(fingerprint_dict))

    """Create a dataset and split it into train_validation data and test data."""
    dataset = list(zip(Smiles, Molecules, adjacencies , correct_properties))#docking_scores
    dataset = list(enumerate(dataset)) 

    train_valid_data =[]
    test_data = []
    train_valid_data = dataset
    index_train_valid, train_valid_data = map(list, zip(*train_valid_data))
    
    """Start training with all data (training + validation)"""
    torch.manual_seed(7688)
    
    def pad(matrices, pad_value):
        """Pad adjacency matrices for batch processing."""
        block_indexs = []
        sizes = [d.shape[0] for d in matrices]
        D = sum(sizes)
        pad_matrices = pad_value + np.zeros((D, D))
        m = 0
        for i, d in enumerate(matrices):
            s_i = sizes[i]
            #fill in the pad_matrics, block by block
            pad_matrices[m:m+s_i, m:m+s_i] = d
            block_indexs.append((m,m+s_i))
            m += s_i
        return torch.FloatTensor(pad_matrices).to(device),block_indexs
    
    dataset = train_valid_data
    print(len(dataset),len(dataset[0]),dataset[0])
    #######################################################################
    dataset = dataset
    N = len(dataset)
    #aux_ = 0
    #Lyn 553
    #Lck 239
    #Src 499
    for i in range(0, N, 499):
            #print('New batch~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            data_batch = list(zip(*dataset[i:i+499]))
            inputs = data_batch[:-1]
            correct_properties = data_batch[-1]
            Smiles, fingerprints_, adjacencies_ = inputs
            adjacencies,block_indexs = pad(adjacencies_, 0)
            #aux_=aux_+1
    adjacencies.cpu().detach().numpy()
    D_incidence = build_edgegraph_incidenceMt(adjacencies.cpu().detach().numpy())

    #######
    rho = 1
    #######
    DTD = torch.matmul(D_incidence.t(),D_incidence).to(device)
    I_DTD_inv = torch.inverse(torch.eye(DTD.shape[0]).to(device) + rho *DTD).to(device)
    #print('I_DTD_inv:',torch.max(I_DTD_inv) , torch.min(I_DTD_inv))
    init_emb = nn.Embedding(n_fingerprint, dim).to(device)
    fingerprints = torch.cat(fingerprints_)
    init_emb_ = init_emb(fingerprints)
    beta_init = np.triu(adjacencies.cpu().detach().numpy())
  
    all_edges = []
    edges_freq_dict = {}
   
    for mol_,mol_adj,prop_ in zip(fingerprints_,adjacencies_,correct_properties):
        #print('PROP_:~',prop_)
        if prop_ == 1:
            edges_this_mol = np.transpose(np.nonzero(mol_adj))
            for edge_ in edges_this_mol:
                new_edge = (mol_[edge_[0]].cpu().detach().numpy().item(),mol_[edge_[1]].cpu().detach().numpy().item())
                all_edges.append(new_edge)
          
    for ele in all_edges:
        amount_ = all_edges.count(ele)
        edges_freq_dict[ele]=amount_
 
    freqs_for_mols = []
    for mol_,mol_adj,prop_ in zip(fingerprints_,adjacencies_,correct_properties):
        edges_this_mol = np.transpose(np.nonzero(mol_adj))
        freqs_for_thisone = []
        for edge_ in edges_this_mol:
            edges_this_mol = np.transpose(np.nonzero(mol_adj))
            new_edge = (mol_[edge_[0]].cpu().detach().numpy().item(),mol_[edge_[1]].cpu().detach().numpy().item())
            if new_edge in edges_freq_dict.keys():
                freqs_for_thisone.append(edges_freq_dict[new_edge])
                #print('Output new edge freq:~',edges_freq_dict[new_edge])
            else:
                edges_freq_dict[new_edge] = 1
                freqs_for_thisone.append(edges_freq_dict[new_edge])
        freqs_for_mols.extend(freqs_for_thisone)

    myLinear_last = myLinear_last().to(device)
    model = GraphNeuralNetwork(adjacencies,init_emb).to(device)
    model = GraphNeuralNetwork(adjacencies,init_emb).to(device)
    print('MODEL:~~~~~~')
    print(model)
    beta_intt_indexs = np.nonzero(beta_init) 
    print(adjacencies.shape,beta_intt_indexs,beta_intt_indexs[0].shape)

    #A trained model is needed here for explain. 
    pretrained_model = '/home/wangyang/eclipse-workspace/EGNN_cleaned/fullmodel/noBnoTree/Src/Pre_correct/1PDL1--radius:0--update:sum--output:mean--dim:10--hidden_layer:1--output_layer:2--batch:553--lr:0.01--lr_decay:0.001--decay_interval:999--weight_decay:1e-05--iteration_prune:20--iteration_retrain:2--iteration_whole:20_notree_src2019'
    
    
  
    if os.path.isfile(pretrained_model):
        print("Pre-trained model exists:" + pretrained_model)
        model.load_state_dict(torch.load(pretrained_model,map_location=torch.device('cuda', 0)),strict='False') #param_file
        #base_test_acc = test(model,val_loader,device)
        #print('base finished:~~~~~~~~~~~~~~~~~~~~~')
        #sys.exit()
    else:
        print("Pre-trained model does not exist, so before pruning we have to pre-train a model.")
        sys.exit()   
    
    trainer = Trainer(model,beta_intt_indexs,freqs_for_mols)
    tester = Tester(model) 
    
    """Set a model."""
    random.seed(500)
    x = [random.randint(1,50000000) for i in range(50)] #128
    x=[1] #To have multiple runs and to average the results. 
    for l in x:
        torch.manual_seed(l)
        model = GraphNeuralNetwork(adjacencies,init_emb).to(device)

        if os.path.isfile(pretrained_model):
            print("Pre-trained model exists:" + pretrained_model)
            model.load_state_dict(torch.load(pretrained_model,map_location=torch.device('cuda', 0))) #param_file
        else:
            print("Pre-trained model does not exist, so before pruning we have to pre-train a model.")
            #sys.exit()   
         
        trainer = Trainer(model,beta_intt_indexs,freqs_for_mols)
        tester = Tester(model)

        """Output files."""
        file_AUC_pr = 'fullmodel/noBnoTree/Pruned_results--' +str(l) + setting + '_notree.txt'
        file_AUC_re = 'fullmodel/noBnoTree/Retrain_results--' +str(l) + setting + '_notree.txt'
        file_model = 'fullmodel/noBnoTree/' + str(l) + setting + '_notree_src'
        result = ('Epoch\tTime(sec)\tLoss_Valid\tL1_Valid\tLr\tLip\tAUC_valid\t'
                  'Prec_Valid\t\tRecall_Valid\tF1_Valid\tKappa_Valid\tAcc\tNonzero\tNonzero\n')
       
        with open(file_AUC_pr, 'w') as f:
            f.write(result + '\n')
            print(result)
        with open(file_AUC_re, 'w') as f:
            f.write(result + '\n')
            print(result)    
        
        """Start training."""
        loss_aux = 99999999 
        for epoch in range(1,iteration_whole):
            start = timeit.default_timer()
            #trainer = Trainer(model)
            #tester = Tester(model)
            print('-------------Pruned-scores~~~~~~~~~~~~~~~~~~~~~~~~~~')
            for epoch_prune in range(1, iteration_prune+1):             
                if epoch_prune % decay_interval == 0:
                    #Yang, I dont let optimizer to set() anyway
                    print('Prune Epoch:~',epoch_prune)
                    trainer.optimizer.param_groups[0]['lr'] *= lr_decay
                else:
                    pass
                loss,l1_loss = trainer.train(dataset,lr_,lip)
               
                AUC, precision, recall, kappa,acc,molecular_vectors,ts = tester.test(dataset)
                F1_score = f1_score(precision, recall)
            
                end = timeit.default_timer()
                time = end - start
                shape_norm = check_network_fussedlasso(model) #return Nonzero_, F_norm, Nonzero_ww
            
                tester.result_AUC(epoch_prune, time, loss,l1_loss, AUC,
                                            precision, recall, F1_score, kappa, acc, file_AUC_pr, shape_norm[0].cpu().detach().numpy(),shape_norm[-1].cpu().detach().numpy(),lr_,lip)
           
            result = [epoch, time, loss,AUC,
                  precision, recall, F1_score, kappa,acc]
            print('\t'.join(map(str, result)))
          
            shape_norm = check_network_fussedlasso(model)
            print('Sasasa:~',shape_norm)
        
            print('-------------Re-train-scores~~~~~~~~~~~~~~~~~~~~~~~~~~')
            for epoch_retrain in range(1,iteration_retrain+1):
                if epoch_retrain % decay_interval == 0:
                    print('Epoch Retrain:~',epoch_retrain)
                    trainer.optimizer.param_groups[0]['lr'] *= lr_decay

                loss_re= trainer.retrain(dataset,lr,reg_decay=1e-3)
                                
                AUC, precision, recall, kappa,acc,molecular_vectors,ts = tester.test(dataset)
                F1_score = f1_score(precision, recall)
            
                end = timeit.default_timer()
                time = end - start
                shape_norm = check_network_fussedlasso(model)
                l1_loss_re = 'Pass'
                tester.result_AUC(epoch_retrain, time, loss_re,l1_loss_re, AUC,
                                            precision, recall, F1_score, kappa, acc, file_AUC_re, shape_norm[0].cpu().detach().numpy(),shape_norm[-1].cpu().detach().numpy(),lr_,lip)
            result = [epoch, time, loss_re,AUC,
                  precision, recall, F1_score, kappa,acc]
            print('\t'.join(map(str, result)))   
            
            
        tester.save_model(model, file_model+str(epoch_prune)+str(epoch))
        
        end_time_ = datetime.now()
        
        time_difference = (end_time_ - start_time_).total_seconds() 
        
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Training Finished:~",time_difference)
        #sys.exit()
        
        
        
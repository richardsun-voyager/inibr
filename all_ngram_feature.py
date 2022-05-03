from torch.nn import utils as nn_utils
import torch
from torch import optim
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.init as init
import scipy.special

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
bound = 1
hardtanh = torch.nn.Hardtanh(-bound, bound)
def get_higher_degree_ngram(inputs, model1, degree=2, hidden_dim=128):
    
    model = model1.cpu()
    inputs = torch.LongTensor(inputs).cpu()
    max_len = inputs.shape[1]
    embs = model.embeddings(inputs).cpu()
    
    #print(embs.shape)
    weight_ih = model.rnn.f_cell.weight_ih.cpu()
    weight_hh = model.rnn.f_cell.linear_hh.weight.cpu()
    gi = F.linear(embs, weight_ih).cpu()
    g = torch.tanh(gi)
    phi = torch.zeros(max_len, max_len, hidden_dim)#.to(device)
    f = {}
    f[0] = 1 - g**2
    f[1] = -g*f[0]
    
    f[2] = -f[0]**2/6+g**2*f[0]/6
    
    f[3] = g*(f[0])**2/18-g**3*f[0]/72
    
    for t in range(max_len):
        phi[t, t] = g[0, t]
        if t == 0:
            continue
        elif t==1:
            theta = F.linear(phi[:t,  :t], weight_hh)
            for i in range(t):
                temp = 0
                for z in range(1, degree+1):
                    temp += f[z-1][0, t]*theta[i, t-1]**z
                phi[i, t] += temp#hardtanh(temp)
            continue
            
        else:
            theta = F.linear(phi[:t,  :t], weight_hh)#phi 0:t-1
            #print(theta.shape)
            for i in range(t):#t>1
                temp = 0
                for z in range(1, degree+1):
                    for j in range(z):
                        if i == t-1 and j>0:
                            continue
                        if i == t-1:
                            #print(i, t-1)
                            phi_ijz = theta[i, t-1]**(z-j)
                        else:
                            phi_ijz = theta[i, t-1]**(z-j) * theta[(i+1):, t-1].sum(0)**j
                        #print(phi_ijz.shape)
                        #print(i, z, j)
                        #phi_ijz = hardtanh(phi_ijz)
                        temp += scipy.special.binom(z, j)*f[z-1][0, t]*phi_ijz
                phi[i, t] += temp# hardtanh(temp)
    scores = model.decoder(phi).squeeze(-1).cpu().detach()         
    return phi, scores

def rnn_phrase_polarity_new(sent, model, hidden_dim=1024):
  '''
  sent: token ids, a list
  '''
  assert len(sent) > 0
  best_model = model.cpu()
  best_model.eval()
  x = best_model.embeddings(torch.LongTensor(sent)).cpu()#.to(device))
  gi = F.linear(x, best_model.rnn.f_cell.weight_ih).cpu()
  f_w = best_model.rnn.f_cell.linear_hh.weight.cpu()
  max_len = len(gi)
  f_inter = torch.tanh(gi)#gi - gi**3/3#
  f_inter_dev = 1 - f_inter**2
  if len(sent) == 1:
    return best_model.decoder(f_inter)[0].item()
  #A = torch.ones(hidden_dim).type_as(gi)
  #O = torch.zeros_like(A)
  J = f_inter_dev.expand(hidden_dim,  
                      1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) * f_w\

#   J = J + A
  #print(max_len)
  def matrix_mat(i, j):
      assert i<j
      temp = J[0, j]   
      for k in reversed(range(i+1, j)):
        temp = temp.matmul(J[0, k])
      return temp
  
  temp = matrix_mat(0, max_len-1)
  #print(temp[0])
  subinfo = temp.matmul(f_inter[0].unsqueeze(1))
  #rint(subinfo.shape)
  subscore = best_model.decoder(subinfo.transpose(0, 1))[0]
  return subscore.item()

def rnn_phrase_polarity_multi(sent, model, hidden_dim=1024):
  '''
  sent: token ids, a list
  '''
  assert len(sent) > 0
  best_model = model.cpu()
  best_model.eval()
  x = best_model.embeddings(torch.LongTensor(sent)).cpu()#.to(device))
  gi = F.linear(x, best_model.rnn.f_cell.weight_ih).cpu()
  f_w = best_model.rnn.f_cell.linear_hh.weight.cpu()
  max_len = len(gi)
  f_inter = torch.tanh(gi)#gi - gi**3/3#
  f_inter_dev = 1 - f_inter**2
  if len(sent) == 1:
    return best_model.decoder(f_inter)[0]
  #A = torch.ones(hidden_dim).type_as(gi)
  #O = torch.zeros_like(A)
  J = f_inter_dev.expand(hidden_dim,  
                      1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) * f_w\

#   J = J + A
  #print(max_len)
  def matrix_mat(i, j):
      assert i<j
      temp = J[0, j]   
      for k in reversed(range(i+1, j)):
        temp = temp.matmul(J[0, k])
      return temp
  
  temp = matrix_mat(0, max_len-1)
  #print(temp[0])
  subinfo = temp.matmul(f_inter[0].unsqueeze(1))
  #rint(subinfo.shape)
  subscore = best_model.decoder(subinfo.transpose(0, 1))[0]
  return subscore.detach()

def rnn_overall_polarity_new(sent, model1, hidden_dim = 512):
    '''
    sent: token ids, torch tensor
    '''
    assert len(sent) > 0
    model = model1.cpu()
    x = model.embeddings(torch.LongTensor(sent)).cpu()#.to(device))
    gi = F.linear(x, model.rnn.f_cell.weight_ih).cpu()
    f_w = model.rnn.f_cell.linear_hh.weight.cpu()
    max_len = len(gi)
    f_inter = torch.tanh(gi)#gi - gi**3/3#
    f_inter_dev = 1 - f_inter**2
    if len(sent) == 1:
        return model.decoder(f_inter)[0].item()
    #A = torch.ones(hidden_dim).type_as(gi)
    #O = torch.zeros_like(A)
    J =f_inter_dev.expand(hidden_dim,  
                      1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) * f_w\

    #J = J + A
    
    polarity_scores = {}   
    original_scores = []
    accumulated_scores = []
    def matrix_mat(i, j):
      assert i<j
      temp = J[0, j]   
      for k in reversed(range(i+1, j)):
        temp = temp.matmul(J[0, k])
      return temp

    for j in range(max_len-1, max_len):
        score = model.decoder(f_inter[j])
        #print('Original', score)
        original_scores.append(score)
        polarity_scores[j] = [score]
        for i in reversed(range(j)):
            #batch_size, hidden_dim, hidden_dim
            temp = matrix_mat(i, j)
            #print(temp.norm(2))
            subinfo = temp.matmul(f_inter[i].unsqueeze(1))
            subscore = model.decoder(subinfo.transpose(0, 1))
            polarity_scores[j].append(subscore[0])
        score = sum(polarity_scores[j])
        accumulated_scores.append(score)
    #print(polarity_scores[max_len-1])
    overall_score = torch.stack(polarity_scores[max_len-1], 0).sum()
    return overall_score.item()


def rnn_overall_polarity_multi(sent, model1, hidden_dim = 512):
    '''
    sent: token ids, torch tensor
    '''
    assert len(sent) > 0
    model = model1.cpu()
    x = model.embeddings(torch.LongTensor(sent)).cpu()#.to(device))
    gi = F.linear(x, model.rnn.f_cell.weight_ih).cpu()
    f_w = model.rnn.f_cell.linear_hh.weight.cpu()
    max_len = len(gi)
    f_inter = torch.tanh(gi)#gi - gi**3/3#
    f_inter_dev = 1 - f_inter**2
    if len(sent) == 1:
        return model.decoder(f_inter)[0] 
    #A = torch.ones(hidden_dim).type_as(gi)
    #O = torch.zeros_like(A)
    J =f_inter_dev.expand(hidden_dim,  
                      1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) * f_w\

    #J = J + A
    
    polarity_scores = {}   
    original_scores = []
    accumulated_scores = []
    def matrix_mat(i, j):
      assert i<j
      temp = J[0, j]   
      for k in reversed(range(i+1, j)):
        temp = temp.matmul(J[0, k])
      return temp

    for j in range(max_len-1, max_len):
        score = model.decoder(f_inter[j])
        #print('Original', score)
        original_scores.append(score)
        polarity_scores[j] = [score]
        for i in reversed(range(j)):
            #batch_size, hidden_dim, hidden_dim
            temp = matrix_mat(i, j)
            #print(temp.norm(2))
            subinfo = temp.matmul(f_inter[i].unsqueeze(1))
            subscore = model.decoder(subinfo.transpose(0, 1))
            polarity_scores[j].append(subscore[0])
        score = sum(polarity_scores[j])
        accumulated_scores.append(score)
    #print(polarity_scores[max_len-1])
    overall_score = torch.stack(polarity_scores[max_len-1], 0)
    #print(overall_score.shape)
    return overall_score.sum(0).detach()

def gru_phrase_polarity_new(sent, model, hidden_dim=1024):
  '''
  sent: token ids, a list
  '''
  assert len(sent) > 0
  best_model = model.cpu()
  best_model.eval()
  x = best_model.embeddings(torch.LongTensor(sent)).cpu()#.to(device))
  gi = F.linear(x, best_model.rnn.f_cell.weight_ih).cpu()
  i_r, i_i, i_n = gi.chunk(3, 1)
  f_r_w = best_model.rnn.f_cell.linear_hh.weight[:hidden_dim].cpu()
  f_z_w = best_model.rnn.f_cell.linear_hh.weight[hidden_dim:(hidden_dim*2)].cpu()
  f_n_w = best_model.rnn.f_cell.linear_hh.weight[(hidden_dim*2):].cpu()
  max_len = len(i_r)
  f_inter = torch.tanh(i_n)*(0.5-0.5*torch.tanh(i_i/2))# - i_n**3/6# 
  fx_1 = 0.5+0.5*torch.tanh(i_r/2)
  fx_2 = 0.25*(1-torch.tanh(i_r/2)**2)
  fx_3 = 0.5+0.5*torch.tanh(i_i/2)
  fx_4 = 0.25*(1-torch.tanh(i_i/2)**2)
  fx_5 = torch.tanh(i_n)
  fx_6 = 1-torch.tanh(i_n)**2
  if len(sent) == 1:
    return best_model.decoder(f_inter)[0].item()

  temp1 = (1-fx_3)*fx_6*fx_1
  temp2 =  - fx_4*fx_5
  temp3 = fx_3
  #O = torch.zeros_like(A)
  J = temp1.expand(hidden_dim,  
                      1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) * f_n_w\
      +temp2.expand(hidden_dim,  
                    1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) * f_z_w\
      +temp3.expand(hidden_dim,  
                      1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) *  torch.eye(hidden_dim).type_as(i_r)

  #print(max_len)
  def matrix_mat(i, j):
      assert i<j
      temp = J[0, j]   
      for k in reversed(range(i+1, j)):
        temp = temp.matmul(J[0, k])
      return temp
  
  temp = matrix_mat(0, max_len-1)
  #print(temp[0])
  subinfo = temp.matmul(f_inter[0].unsqueeze(1))
  #rint(subinfo.shape)
  subscore = best_model.decoder(subinfo.transpose(0, 1))[0]
  return subscore.item()

def gru_overall_polarity_new(sent, model1, hidden_dim = 512):
    '''
    sent: token ids, torch tensor 
    '''
    assert len(sent) > 0
    model = model1.cpu()
    x = model.embeddings(torch.LongTensor(sent)).cpu()#.to(device))
    gi = F.linear(x, model.rnn.f_cell.weight_ih).cpu()
    i_r, i_i, i_n = gi.chunk(3, 1)
    f_r_w = model.rnn.f_cell.linear_hh.weight[:hidden_dim].cpu()
    f_z_w = model.rnn.f_cell.linear_hh.weight[hidden_dim:(hidden_dim*2)].cpu()
    f_n_w = model.rnn.f_cell.linear_hh.weight[(hidden_dim*2):].cpu()
    max_len = len(i_r)
    
    fx_1 = 0.5+0.5*torch.tanh(i_r/2)
    fx_2 = 0.25*(1-torch.tanh(i_r/2)**2)
    fx_3 = 0.5+0.5*torch.tanh(i_i/2)
    fx_4 = 0.25*(1-torch.tanh(i_i/2)**2)
    fx_5 = torch.tanh(i_n)
    fx_6 = 1-torch.tanh(i_n)**2
    f_inter = fx_5*(1-fx_3)# - i_n**3/6# 
    if len(sent) == 1:
        return model.decoder(f_inter)[0].item()

    temp1 = (1-fx_3)*fx_6*fx_1
    temp2 =  - fx_4*fx_5
    temp3 = fx_3
    #O = torch.zeros_like(A)
    J = temp1.expand(hidden_dim,  
                      1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) * f_n_w\
      +temp2.expand(hidden_dim,  
                    1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) * f_z_w\
      +temp3.expand(hidden_dim,  
                      1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) *  torch.eye(hidden_dim).type_as(i_r)
    
    polarity_scores = {}   
    original_scores = []
    accumulated_scores = []
    def matrix_mat(i, j):
      assert i<j
      temp = J[0, j]   
      for k in reversed(range(i+1, j)):
        temp = temp.matmul(J[0, k])
      return temp

    for j in range(max_len-1, max_len):
        score = model.decoder(f_inter[j])
        #print('Original', score)
        original_scores.append(score)
        polarity_scores[j] = [score]
        for i in reversed(range(j)):
            #batch_size, hidden_dim, hidden_dim
            temp = matrix_mat(i, j)
            #print(temp.norm(2))
            subinfo = temp.matmul(f_inter[i].unsqueeze(1))
            subscore = model.decoder(subinfo.transpose(0, 1))
            polarity_scores[j].append(subscore[0])
        score = sum(polarity_scores[j])
        accumulated_scores.append(score)
    #print(polarity_scores[max_len-1])
    overall_score = torch.stack(polarity_scores[max_len-1], 0).sum()
    return overall_score.item()

def gru_phrase_polarity_multi(sent, model1, hidden_dim = 512):
    '''
    sent: token ids, torch tensor
    '''
    assert len(sent) > 0
    model = model1.cpu()
    x = model.embeddings(torch.LongTensor(sent)).cpu()#.to(device))
    gi = F.linear(x, model.rnn.f_cell.weight_ih).cpu()
    i_r, i_i, i_n = gi.chunk(3, 1)
    f_r_w = model.rnn.f_cell.linear_hh.weight[:hidden_dim].cpu()
    f_z_w = model.rnn.f_cell.linear_hh.weight[hidden_dim:(hidden_dim*2)].cpu()
    f_n_w = model.rnn.f_cell.linear_hh.weight[(hidden_dim*2):].cpu()
    max_len = len(i_r)
 
    fx_1 = 0.5+0.5*torch.tanh(i_r/2)
    fx_2 = 0.25*(1-torch.tanh(i_r/2)**2)
    fx_3 = 0.5+0.5*torch.tanh(i_i/2)
    fx_4 = 0.25*(1-torch.tanh(i_i/2)**2)
    fx_5 = torch.tanh(i_n)
    fx_6 = 1-torch.tanh(i_n)**2
    f_inter = fx_5*(1-fx_3)# - i_n**3/6# 
    if len(sent) == 1:
        return model.decoder(f_inter)[0]#.item()

    temp1 = (1-fx_3)*fx_6*fx_1
    temp2 =  - fx_4*fx_5
    temp3 = fx_3
    #O = torch.zeros_like(A)
    J = temp1.expand(hidden_dim,  
                      1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) * f_n_w\
      +temp2.expand(hidden_dim,  
                    1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) * f_z_w\
      +temp3.expand(hidden_dim,  
                      1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) *  torch.eye(hidden_dim).type_as(i_r)
    
    
    def matrix_mat(i, j):
      assert i<j
      temp = J[0, j]   
      for k in reversed(range(i+1, j)):
        temp = temp.matmul(J[0, k])
      return temp

    temp = matrix_mat(0, max_len-1)
    subinfo = temp.matmul(f_inter[0].unsqueeze(1))

    #print(subinfo.shape)
    subscore = model.decoder(subinfo.transpose(0, 1))[0].detach()
    return subscore


def gru_overall_polarity_multi(sent, model1, hidden_dim = 512):
    '''
    sent: token ids, torch tensor
    '''
    assert len(sent) > 0
    model = model1.cpu()
    x = model.embeddings(torch.LongTensor(sent)).cpu()#.to(device))
    gi = F.linear(x, model.rnn.f_cell.weight_ih).cpu()
    i_r, i_i, i_n = gi.chunk(3, 1)
    f_r_w = model.rnn.f_cell.linear_hh.weight[:hidden_dim].cpu()
    f_z_w = model.rnn.f_cell.linear_hh.weight[hidden_dim:(hidden_dim*2)].cpu()
    f_n_w = model.rnn.f_cell.linear_hh.weight[(hidden_dim*2):].cpu()
    max_len = len(i_r)
  
    fx_1 = 0.5+0.5*torch.tanh(i_r/2)
    fx_2 = 0.25*(1-torch.tanh(i_r/2)**2)
    fx_3 = 0.5+0.5*torch.tanh(i_i/2)
    fx_4 = 0.25*(1-torch.tanh(i_i/2)**2)
    fx_5 = torch.tanh(i_n)
    fx_6 = 1-torch.tanh(i_n)**2
    f_inter = fx_5*(1-fx_3)# - i_n**3/6# 
    if len(sent) == 1:
        return model.decoder(f_inter)[0]#.item()

    temp1 = (1-fx_3)*fx_6*fx_1
    temp2 =  - fx_4*fx_5
    temp3 = fx_3
    #O = torch.zeros_like(A)
    J = temp1.expand(hidden_dim,  
                      1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) * f_n_w\
      +temp2.expand(hidden_dim,  
                    1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) * f_z_w\
      +temp3.expand(hidden_dim,  
                      1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) *  torch.eye(hidden_dim).type_as(i_r)
    #print('max j:', J.abs().max())
    
    def matrix_mat(i, j):
      assert i<j
      temp = J[0, j]   
      for k in reversed(range(i+1, j)):
        temp = temp.matmul(J[0, k])
      return temp
    current_score = model.decoder(f_inter)[max_len-1].detach()
    nested_score = 0
    for k in range(max_len-1):
      temp = matrix_mat(k, max_len-1)
      subinfo = temp.matmul(f_inter[k].unsqueeze(1))
      nested_score += model.decoder(subinfo.transpose(0, 1))[0].detach()
    return nested_score + current_score

def lstm_phrase_polarity_new(sent, model, hidden_dim=1024):
    '''
    sent: token ids, a list
    '''
    assert len(sent) > 0
    best_model = model.cpu()
    x = best_model.embeddings(torch.LongTensor(sent)).cpu()#.to(device))
    gi = F.linear(x, best_model.rnn.f_cell.weight_ih).cpu()
    i_i, i_f, i_g, i_o = gi.chunk(4, 1)
    f_i_w = best_model.rnn.f_cell.linear_hh.weight[:hidden_dim].cpu()
    f_f_w = best_model.rnn.f_cell.linear_hh.weight[hidden_dim:(hidden_dim*2)].cpu()
    f_g_w = best_model.rnn.f_cell.linear_hh.weight[(hidden_dim*2):(hidden_dim*3)].cpu()
    f_o_w = best_model.rnn.f_cell.linear_hh.weight[(hidden_dim*3):].cpu()
    max_len = len(i_i)

    gx_i = torch.sigmoid(i_i)#0.5 + torch.tanh(i_i/2)/2
    fx_i =  torch.sigmoid(i_i)*(1- torch.sigmoid(i_i))#0.25*(1-torch.tanh(i_i/2)**2)
    gx_f = torch.sigmoid(i_f)#0.5 + torch.tanh(i_f/2)/2
    fx_f = torch.sigmoid(i_f)*(1- torch.sigmoid(i_f))#0.25*(1-torch.tanh(i_f/2)**2)
    gx_o = torch.sigmoid(i_o)#0.5+torch.tanh(i_o/2)/2
    fx_o = torch.sigmoid(i_o)*(1- torch.sigmoid(i_o))#0.25*(1-torch.tanh(i_o/2)**2)
    gx_g = torch.tanh(i_g)
    fx_g = 1-torch.tanh(i_g)**2
    
    f_c = gx_i * gx_g
    f_h = gx_o * torch.tanh(f_c)
    if len(sent) == 1:
        return best_model.decoder(f_h)[0].item()
    B = gx_f.expand(hidden_dim,  
                    1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) * torch.eye(hidden_dim).type_as(i_i)
    #O = torch.zeros_like(A)
    D = (gx_g*fx_i).expand(hidden_dim,  
                      1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) * f_i_w\
    +(gx_i*fx_g).expand(hidden_dim,  
                      1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) * f_g_w
    
    E = (gx_o*(1-torch.tanh(f_c)**2)).expand(hidden_dim,  
                      1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) * B

    FF = (gx_o*(1-torch.tanh(f_c)**2)).expand(hidden_dim,  
                      1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) * D\
    +(fx_o*torch.tanh(f_c)).expand(hidden_dim,  
                      1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) * f_o_w
    row1 = torch.cat([B, D], 3)
    row2 = torch.cat([E, FF], 3) 
    J = torch.cat([row1, row2], 2)

    #print(max_len)
    def matrix_mat(i, j):
      assert i<j
      temp = J[0, j]   
      for k in reversed(range(i+1, j)):
        temp = temp.matmul(J[0, k])
      return temp

    temp = matrix_mat(0, max_len-1)
    #print(temp.shape)
    f = torch.cat([f_c, f_h], 1)
    #print(f.shape)
    subinfo = temp.matmul(f[0].unsqueeze(1))[hidden_dim:]
    #rint(subinfo.shape)
    subscore = best_model.decoder(subinfo.transpose(0, 1))[0]
    return subscore.cpu().item()

def lstm_overall_polarity_new(sent, model, hidden_dim=1024):
    '''
    sent: token ids, a list
    '''
    assert len(sent) > 0
    best_model = model.cpu()
    x = best_model.embeddings(torch.LongTensor(sent)).cpu()#.to(device))
    gi = F.linear(x, best_model.rnn.f_cell.weight_ih).cpu()
    i_i, i_f, i_g, i_o = gi.chunk(4, 1)
    f_i_w = best_model.rnn.f_cell.linear_hh.weight[:hidden_dim].cpu()
    f_f_w = best_model.rnn.f_cell.linear_hh.weight[hidden_dim:(hidden_dim*2)].cpu()
    f_g_w = best_model.rnn.f_cell.linear_hh.weight[(hidden_dim*2):(hidden_dim*3)].cpu()
    f_o_w = best_model.rnn.f_cell.linear_hh.weight[(hidden_dim*3):].cpu()
    max_len = len(i_i)

    gx_i = torch.sigmoid(i_i)#0.5 + torch.tanh(i_i/2)/2
    fx_i =  torch.sigmoid(i_i)*(1- torch.sigmoid(i_i))#0.25*(1-torch.tanh(i_i/2)**2)
    gx_f = torch.sigmoid(i_f)#0.5 + torch.tanh(i_f/2)/2
    fx_f = torch.sigmoid(i_f)*(1- torch.sigmoid(i_f))#0.25*(1-torch.tanh(i_f/2)**2)
    gx_o = torch.sigmoid(i_o)#0.5+torch.tanh(i_o/2)/2
    fx_o = torch.sigmoid(i_o)*(1- torch.sigmoid(i_o))#0.25*(1-torch.tanh(i_o/2)**2)
    gx_g = torch.tanh(i_g)
    fx_g = 1-torch.tanh(i_g)**2
    
    
    f_c = gx_i * gx_g
    f_h = gx_o * torch.tanh(f_c)
    if len(sent) == 1:
        return best_model.decoder(f_h)[0].item()
    B = gx_f.expand(hidden_dim,  
                    1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) * torch.eye(hidden_dim).type_as(i_i)
    #O = torch.zeros_like(A)
    D = (gx_g*fx_i).expand(hidden_dim,  
                      1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) * f_i_w\
    +(gx_i*fx_g).expand(hidden_dim,  
                      1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) * f_g_w
    
    E = (gx_o*(1-torch.tanh(f_c)**2)).expand(hidden_dim,  
                      1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) * B

    FF = (gx_o*(1-torch.tanh(f_c)**2)).expand(hidden_dim,  
                      1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) * D\
    +(fx_o*torch.tanh(f_c)).expand(hidden_dim,  
                      1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) * f_o_w
    
    row1 = torch.cat([B, D], 3)
    row2 = torch.cat([E, FF], 3) 
    J = torch.cat([row1, row2], 2)
    
    polarity_scores = {}   
    original_scores = []
    accumulated_scores = []
    def matrix_mat(i, j):
      assert i<j
      temp = J[0, j]   
      for k in reversed(range(i+1, j)):
        temp = temp.matmul(J[0, k])
      return temp

    f = torch.cat([f_c, f_h], 1)

    for j in range(max_len-1, max_len):
        score = best_model.decoder(f_h[j])
        #print('Original', score)
        original_scores.append(score.item())
        polarity_scores[j] = [score.item()]
        for i in reversed(range(j)):
            #batch_size, hidden_dim, hidden_dim
            temp = matrix_mat(i, j)
            #print(temp.norm(2))

            #print(f.shape)
            subinfo = temp.matmul(f[i].unsqueeze(1))
            #print(subinfo.abs().max().item(), j, i)
            subinfo_h = subinfo[hidden_dim:]
            #subinfo = temp.matmul(f_inter[i].unsqueeze(1))
            subscore = best_model.decoder(subinfo_h.transpose(0, 1))
            polarity_scores[j].append(subscore.item())
        score = sum(polarity_scores[j])
        accumulated_scores.append(score)

    #print(polarity_scores[j])
    return score

def lstm_phrase_polarity_multi(sent, model, hidden_dim=512):
    best_model = model.cpu()
    x = best_model.embeddings(torch.LongTensor(sent)).cpu()#.to(device))
    gi = F.linear(x, best_model.rnn.f_cell.weight_ih).cpu()
    i_i, i_f, i_g, i_o = gi.chunk(4, 1)
    f_i_w = best_model.rnn.f_cell.linear_hh.weight[:hidden_dim].cpu()
    f_f_w = best_model.rnn.f_cell.linear_hh.weight[hidden_dim:(hidden_dim*2)].cpu()
    f_g_w = best_model.rnn.f_cell.linear_hh.weight[(hidden_dim*2):(hidden_dim*3)].cpu()
    f_o_w = best_model.rnn.f_cell.linear_hh.weight[(hidden_dim*3):].cpu()
    max_len = len(i_i)
    #print(max_len)

    gx_i = torch.sigmoid(i_i)#0.5 + torch.tanh(i_i/2)/2
    fx_i =  torch.sigmoid(i_i)*(1- torch.sigmoid(i_i))#0.25*(1-torch.tanh(i_i/2)**2)
    gx_f = torch.sigmoid(i_f)#0.5 + torch.tanh(i_f/2)/2
    fx_f = torch.sigmoid(i_f)*(1- torch.sigmoid(i_f))#0.25*(1-torch.tanh(i_f/2)**2)
    gx_o = torch.sigmoid(i_o)#0.5+torch.tanh(i_o/2)/2
    fx_o = torch.sigmoid(i_o)*(1- torch.sigmoid(i_o))#0.25*(1-torch.tanh(i_o/2)**2)
    gx_g = torch.tanh(i_g)
    fx_g = 1-torch.tanh(i_g)**2
    
    
    f_c = gx_i * gx_g
    f_h = gx_o * torch.tanh(f_c)
    if len(sent) == 1:
        return best_model.decoder(f_h)[0]
    B = gx_f.expand(hidden_dim,  
                    1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) * torch.eye(hidden_dim).type_as(i_i)
    #O = torch.zeros_like(A)
    D = (gx_g*fx_i).expand(hidden_dim,  
                      1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) * f_i_w\
    +(gx_i*fx_g).expand(hidden_dim,  
                      1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) * f_g_w
    
    E = (gx_o*(1-torch.tanh(f_c)**2)).expand(hidden_dim,  
                      1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) * B

    FF = (gx_o*(1-torch.tanh(f_c)**2)).expand(hidden_dim,  
                      1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) * D\
    +(fx_o*torch.tanh(f_c)).expand(hidden_dim,  
                      1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) * f_o_w
    
    row1 = torch.cat([B, D], 3)
    row2 = torch.cat([E, FF], 3) 
    J = torch.cat([row1, row2], 2)
    
    def matrix_mat(i, j):
      assert i<j
      temp = J[0, j]   
      for k in reversed(range(i+1, j)):
        temp = temp.matmul(J[0, k])
      return temp

    f = torch.cat([f_c, f_h], 1)
    
    polarity_scores = {}   
    original_scores = []
    accumulated_scores = []

    for j in range(max_len-1, max_len):
        score = best_model.decoder(f_h[j])
        #print('Original', score)
        original_scores.append(score)
        polarity_scores[j] = [score]
        for i in reversed(range(j)):
            #batch_size, hidden_dim, hidden_dim
            temp = matrix_mat(i, j)
            #print(temp.norm(2))

            #print(f.shape)
            subinfo = temp.matmul(f[i].unsqueeze(1))
            #print(subinfo.abs().max().item(), j, i)
            subinfo_h = subinfo[hidden_dim:]
            #subinfo = temp.matmul(f_inter[i].unsqueeze(1))
            subscore = best_model.decoder(subinfo_h.transpose(0, 1))
            polarity_scores[j].append(subscore[0])
            
        score = sum(polarity_scores[j])
        accumulated_scores.append(score)
    return subscore[0].detach()

def lstm_overall_polarity_multi(sent, model, hidden_dim=1024):
    '''
    sent: token ids, a list
    '''
    assert len(sent) > 0
    best_model = model.cpu()
    x = best_model.embeddings(torch.LongTensor(sent)).cpu()#.to(device))
    gi = F.linear(x, best_model.rnn.f_cell.weight_ih).cpu()
    i_i, i_f, i_g, i_o = gi.chunk(4, 1)
    f_i_w = best_model.rnn.f_cell.linear_hh.weight[:hidden_dim].cpu()
    f_f_w = best_model.rnn.f_cell.linear_hh.weight[hidden_dim:(hidden_dim*2)].cpu()
    f_g_w = best_model.rnn.f_cell.linear_hh.weight[(hidden_dim*2):(hidden_dim*3)].cpu()
    f_o_w = best_model.rnn.f_cell.linear_hh.weight[(hidden_dim*3):].cpu()
    max_len = len(i_i)

    gx_i = torch.sigmoid(i_i)#0.5 + torch.tanh(i_i/2)/2
    fx_i =  torch.sigmoid(i_i)*(1- torch.sigmoid(i_i))#0.25*(1-torch.tanh(i_i/2)**2)
    gx_f = torch.sigmoid(i_f)#0.5 + torch.tanh(i_f/2)/2
    fx_f = torch.sigmoid(i_f)*(1- torch.sigmoid(i_f))#0.25*(1-torch.tanh(i_f/2)**2)
    gx_o = torch.sigmoid(i_o)#0.5+torch.tanh(i_o/2)/2
    fx_o = torch.sigmoid(i_o)*(1- torch.sigmoid(i_o))#0.25*(1-torch.tanh(i_o/2)**2)
    gx_g = torch.tanh(i_g)
    fx_g = 1-torch.tanh(i_g)**2
    
    
    f_c = gx_i * gx_g
    f_h = gx_o * torch.tanh(f_c)
    if len(sent) == 1:
        return best_model.decoder(f_h)[0]
    B = gx_f.expand(hidden_dim,  
                    1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) * torch.eye(hidden_dim).type_as(i_i)
    #O = torch.zeros_like(A)
    D = (gx_g*fx_i).expand(hidden_dim,  
                      1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) * f_i_w\
    +(gx_i*fx_g).expand(hidden_dim,  
                      1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) * f_g_w
    
    E = (gx_o*(1-torch.tanh(f_c)**2)).expand(hidden_dim,  
                      1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) * B

    FF = (gx_o*(1-torch.tanh(f_c)**2)).expand(hidden_dim,  
                      1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) * D\
    +(fx_o*torch.tanh(f_c)).expand(hidden_dim,  
                      1, max_len, 
                      hidden_dim).transpose(0, 1).transpose(1, 2).transpose(2, 3) * f_o_w
    
    row1 = torch.cat([B, D], 3)
    row2 = torch.cat([E, FF], 3) 
    J = torch.cat([row1, row2], 2)
    
    polarity_scores = {}   
    original_scores = []
    accumulated_scores = []
    def matrix_mat(i, j):
      assert i<j
      temp = J[0, j]   
      for k in reversed(range(i+1, j)):
        temp = temp.matmul(J[0, k])
      return temp

    f = torch.cat([f_c, f_h], 1)

    for j in range(max_len-1, max_len):
        score = best_model.decoder(f_h[j])
        #print('Original', score)
        original_scores.append(score)
        polarity_scores[j] = [score]
        for i in reversed(range(j)):
            #batch_size, hidden_dim, hidden_dim
            temp = matrix_mat(i, j)
            #print(temp.norm(2))

            #print(f.shape)
            subinfo = temp.matmul(f[i].unsqueeze(1))
            #print(subinfo.abs().max().item(), j, i)
            subinfo_h = subinfo[hidden_dim:]
            #subinfo = temp.matmul(f_inter[i].unsqueeze(1))
            subscore = best_model.decoder(subinfo_h.transpose(0, 1))
            polarity_scores[j].append(subscore)
        score = sum(polarity_scores[j])
        accumulated_scores.append(score)

    #print(polarity_scores[j])
    return score


import torch
import torch.nn as nn
import torch.nn.functional as F









# _____________________________________________________________________________



class Attention(nn.Module):


  def __init__(self,n_embd,n_head) :
    super().__init__()

    self.nh = n_head

    self.w = nn.Linear(n_embd,3*n_embd)    # 3 * n_head * head_size
    self.proj = nn.Linear(n_embd,n_embd)


  def forward(self,x):
    
    B,T,C = x.shape
    
    wei = self.w(x)        # B,T, 3* C 

    k,q,v = torch.chunk(wei,3, dim = -1)      # each B,T,C

    head_size = C//self.nh

    key   = k.view(B, T, self.nh, head_size).transpose(1, 2)    # B, n_head, T, head_size
    query = q.view(B, T, self.nh, head_size).transpose(1, 2)    # ""
    value = v.view(B, T, self.nh, head_size).transpose(1, 2)


    weight = ( query @ key.transpose(-1,-2) )  * (head_size ** -0.5)    #B,nh,T,T
    weight = F.softmax(weight,dim = -1)

    out = weight @ value      #B,nh,T,n_head

    out.transpose(1,2)

    out = self.proj(out.view(B,T,C))

    return out



# _____________________________________________________________________________


class MLP(nn.Module):


  def __init__(self,n_embd):
    super().__init__()


    self.layer = nn.Linear(n_embd,4*n_embd)
    self.gelu = nn.GELU()
    self.proj = nn.Linear(4*n_embd,n_embd)



  def forward(self,x):
    

    x = self.gelu(self.layer(x))
    x = self.proj(x)
    
    return x



# _____________________________________________________________________________




class Block(nn.Module):


  def __init__(self,n_layer,n_embd,n_head):
    super().__init__()


    self.ln_1 = nn.LayerNorm(n_embd)
    self.attn = Attention(n_embd,n_head)
    self.ln_2 = nn.LayerNorm(n_embd)
    self.mlp = MLP(n_embd)


  def forward(self,x):

    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))

    return x



# _____________________________________________________________________________



class PatchEmbd(nn.Module):
    
    def __init__(self):
      super().__init__()
      
      pass
  
  
  
    def forward(self,x):
        pass




# _____________________________________________________________________________


class ViT(nn.Module):
    
    def __init__(self):
      super().__init__()
      
      
      pass
  


    def forward(self,x):
        
        pass 
    
    

# _____________________________________________________________________________
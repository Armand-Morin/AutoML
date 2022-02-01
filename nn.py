import torch
import nni.retiarii.nn.pytorch as nn


from random import randrange, choice


Activation = [nn.ReLU(),nn.LeakyReLU(),nn.Softplus(),nn.Tanh(),nn.Sigmoid()]

OPS = {
    'relu': lambda dim_in , dim_out , bias : Linear_relu(dim_in , dim_out , bias=bias),
    'tanh': lambda dim_in , dim_out , bias : Linear_tanh(dim_in , dim_out , bias=bias),
    'leaky_relu': lambda dim_in , dim_out , bias : Linear_leaky_relu(dim_in , dim_out , bias=bias),
    'soft_plus': lambda dim_in , dim_out , bias : Linear_soft_plus(dim_in , dim_out , bias=bias),
    'sigmoid': lambda dim_in , dim_out , bias : Linear_sigmoid(dim_in , dim_out , bias=bias),
    'aleat': lambda dim_in , dim_out , bias : aleat(dim_in , dim_out , bias=bias)}


class Linear_relu(nn.Module):
    def __init__(self , dim_in , dim_out , bias=True):
        super().__init__()
        self.linear = nn.Linear(dim_in , dim_out , bias=bias)
        self.activation = nn.ReLU()
    
    def forward(self , x):
        out = self.linear(x)
        out = self.activation(out)
        return out

    
class Linear_leaky_relu(nn.Module):
    def __init__(self , dim_in , dim_out , bias=True):
        super().__init__()
        self.linear = nn.Linear(dim_in , dim_out , bias=bias)
        self.activation = nn.LeakyReLU()
    
    def forward(self , x):
        out = self.linear(x)
        out = self.activation(out)
        return out


class Linear_soft_plus(nn.Module):
    def __init__(self , dim_in , dim_out , bias=True):
        super().__init__()
        self.linear = nn.Linear(dim_in , dim_out , bias=bias)
        self.activation = nn.Softplus()
    
    def forward(self , x):
        out = self.linear(x)
        out = self.activation(out)
        return out

    
class Linear_tanh(nn.Module):
    def __init__(self , dim_in , dim_out , bias=True):
        super().__init__()
        self.linear = nn.Linear(dim_in , dim_out , bias=bias)
        self.activation = nn.Tanh()
    
    def forward(self , x):
        out = self.linear(x)
        out = self.activation(out)
        return out

    
class Linear_sigmoid(nn.Module):
    def __init__(self , dim_in , dim_out , bias=True):
        super().__init__()
        self.linear = nn.Linear(dim_in , dim_out , bias=bias)
        self.activation = nn.Sigmoid()
    
    def forward(self , x):
        out = self.linear(x)
        out = self.activation(out)
        return out


class aleat(nn.Module):
    def __init__(self , dim_in , dim_out , bias=True):
        super().__init__()

        self.linear = nn.Linear(dim_in , dim_out , bias=bias)
        self.activation = nn.LayerChoice(Activation)
    
    def forward(self , x):
        out = self.linear(x)
        out = self.activation(out)
        return out


def genrateurneurones(depth):
    # fonction qui renvoi la liste du nombre de neurones choisis aleatoirement pour chaque couche
    L=[]
    for i in range(depth):
        L.append(nn.ValueChoice([16,32,64,128,256]))
    return L


def genrateurcouche(L_couches,depth):
    L=[]
    for i in range(depth):
        L.append(nn.LayerChoice(L_couches))
    return L


#classe qui crée un mélange de toutes les connexions
class MixedOp(nn.Module):
    def __init__(self, dim_in , dim_out,depth):
        super().__init__()
        self._ops = nn.ModuleList()

        L=genrateurneurones(depth)
        self._ops.append(OPS[list(OPS)[randrange(len(OPS))]](dim_in ,L[0],bias=True))

        for x in range(1,depth):
            op = OPS[list(OPS)[randrange(len(OPS))]](L[x-1] , L[x] , bias=True)
            self._ops.append(op)

        self._ops.append(OPS[list(OPS)[randrange(len(OPS))]](L[-1] ,dim_out,bias=True))

    def forward(self, x, weights, mask):
        # weights: weight for each operation
        return sum(w * m * op(x) for m, w, op in zip(mask, weights, self._ops))


class MixedAl(nn.Module):
    def __init__(self, dim_in , dim_out,depth):
        super().__init__()
        self._ops = nn.ModuleList()

        L=genrateurneurones(depth)
        self._ops.append(OPS['aleat'](dim_in ,L[0],bias=True))

        for x in range(1,depth):
            op = OPS['aleat'](L[x-1] , L[x] , bias=True)
            self._ops.append(op)

        self._ops.append(OPS['aleat'](L[-1] ,dim_out,bias=True))

    def forward(self, x, weights, mask):
        # weights: weight for each operation
        return sum(w * m * op(x) for m, w, op in zip(mask, weights, self._ops))

M=MixedAl(dim_in=3,dim_out=1,depth=4)
print(M)
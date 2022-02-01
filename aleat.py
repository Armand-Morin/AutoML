import torch
import nni.retiarii.nn.pytorch as nn

class aleat(nn.Module):
    def __init__(self , dim_in , dim_out , bias=True):
        super().__init__()
        self.Activation = [nn.ReLU(),nn.LeakyReLU(),nn.Softplus(),nn.Tanh(),nn.Sigmoid()]
        self.linear = nn.Linear(dim_in , dim_out , bias=bias)
        self.activation = nn.LayerChoice(self.Activation)
    
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


class MixedAl(nn.Module):
    def __init__(self,depth, delay=None):
        super().__init__()
        self._ops = nn.ModuleList()

        self.in_size = 3
        if delay is not None:
            self.in_size = 4

        L=genrateurneurones(depth)
        self._ops.append(aleat(self.in_size ,L[0],bias=True))

        for x in range(1,depth):
            op = aleat(L[x-1] , L[x] , bias=True)
            self._ops.append(op)

        self._ops.append(aleat(L[-1] ,1,bias=True))

    def forward(self , x):
        for f in self._ops:
            x = f(x)
        return x


model_quant=MixedAl(depth=5, delay=None)
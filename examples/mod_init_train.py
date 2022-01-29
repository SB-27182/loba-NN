# Data
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import numpy as np

# Model:
import torch
from survae.transforms import AffineCouplingBijection, ActNormBijection, Reverse
from survae.transforms import ConditionalAffineCouplingBijection
from survae.nn.layers import ElementwiseParams
# mod:
from survae.distributions.conditional._data_OsciCond_dist import dOscicond_Ovarmean_Nvar_radi_dist
from survae.flows._data_osciCond_flow import dOsciCond_flow

# Utils:
from survae.utils import mod_chooseDim
import random

# Optim
from torch.optim import Adam



sample_observations_count = 988
#------------------------------# Data #-----------------------------#
class trainDataSet(Dataset):
    def __init__(self):
        raw = np.loadtxt("./data/osciCond_data2.csv", delimiter=",", dtype=np.float32)
        x = raw[:, [0]]
        y = raw[:, [1]]
        z = raw[:, [2]]
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.z = torch.from_numpy(z)
        self.data = torch.cat((self.x, self.y, self.z), 1)
        self.n__samples = raw.shape[0]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.n__samples
dataset_tr = trainDataSet()
train_loader = DataLoader(dataset=dataset_tr, batch_size=sample_observations_count, shuffle=True)
test_loader = DataLoader(dataset=dataset_tr, batch_size=sample_observations_count, shuffle=True)



#================================# SubModule NN with dist #==================================#
class dist_net(nn.Module):
    def __init__(self, inDim, outDim):
        super(dist_net, self).__init__()
        self.lin = nn.Linear(inDim, outDim)

        #--------------------------------# Distribution & Sample #----------------------------------#
        dist = torch.distributions.Normal(1.0, 0.01)
        a = dist.sample([outDim, inDim])
        a = a.detach()
        #-------------------------------------------------------------------------------------------#

        self.lin.weight = torch.nn.Parameter(a)


    def forward(self, x):
        return(self.lin(x))





#-----------------------------------------# Model #------------------------------------------------#
def n_var_net():
    net = nn.Sequential(dist_net(25, 200), nn.ReLU(),
                         nn.Linear(200, 100), nn.ReLU(),
                         nn.Linear(100, 100), nn.ReLU(),
                         nn.Linear(100, 50), nn.ReLU(),
                         nn.Linear(50, 2))

    return net

def cupNN():
    return nn.Sequential(nn.Linear(1, 200), nn.ReLU(),
                         nn.Linear(200, 100), nn.ReLU(),
                         nn.Linear(100, 200), nn.ReLU(),
                         nn.Linear(200, 100), nn.ReLU(),
                         nn.Linear(100, 2), ElementwiseParams(2))
def cupcNN():
  return nn.Sequential(nn.Linear(2, 200), nn.ReLU(),
                       nn.Linear(200, 100), nn.ReLU(),
                       nn.Linear(100, 200), nn.ReLU(),
                       nn.Linear(200, 100), nn.ReLU(),
                       nn.Linear(100, 2), ElementwiseParams(2))
def uNN():
    return nn.Sequential(nn.Linear(1, 200), nn.ReLU(),
                         nn.Linear(200, 300), nn.ReLU(),
                         nn.Linear(300, 200), nn.ReLU(),
                         nn.Linear(200, 100), nn.ReLU(),
                         nn.Linear(100, 50), nn.ReLU(),
                         nn.Linear(50, 25))
def concNN():
    return nn.Sequential(nn.Linear(25, 200), nn.ReLU(),
                         nn.Linear(200, 100), nn.ReLU(),
                         nn.Linear(100, 1), ElementwiseParams(1))

def radii_net():
    return nn.Sequential(nn.Linear(25, 200), nn.ReLU(),
                         nn.Linear(200, 100), nn.ReLU(),
                         nn.Linear(100, 100), nn.ReLU(),
                         nn.Linear(100, 50), nn.ReLU(),
                         nn.Linear(50, 2))
def osci_var_net():
    return nn.Sequential(nn.Linear(25, 200), nn.ReLU(),
                         nn.Linear(200, 100), nn.ReLU(),
                         nn.Linear(100, 100), nn.ReLU(),
                         nn.Linear(100, 50), nn.ReLU(),
                         nn.Linear(50, 2))
def o_mean_net():
    return nn.Sequential(nn.Linear(25, 200), nn.ReLU(),
                         nn.Linear(200, 100), nn.ReLU(),
                         nn.Linear(100, 100), nn.ReLU(),
                         nn.Linear(100, 50), nn.ReLU(),
                         nn.Linear(50, 2))



model = dOsciCond_flow(base_dist=dOscicond_Ovarmean_Nvar_radi_dist(radii_net=radii_net(), osci_spread_net=osci_var_net(), n_var_net=n_var_net(), o_mean_net=o_mean_net()),
             transforms=[
               ConditionalAffineCouplingBijection(cupcNN(), context_net=concNN()), ActNormBijection(2), Reverse(2),
               ConditionalAffineCouplingBijection(cupcNN(), context_net=concNN()), ActNormBijection(2), Reverse(2),
               ConditionalAffineCouplingBijection(cupcNN(), context_net=concNN()), ActNormBijection(2), Reverse(2),
               ConditionalAffineCouplingBijection(cupcNN(), context_net=concNN()), ActNormBijection(2), Reverse(2),
               AffineCouplingBijection(cupNN()), ActNormBijection(2), Reverse(2),
               AffineCouplingBijection(cupNN()), ActNormBijection(2), Reverse(2),
               AffineCouplingBijection(cupNN()), ActNormBijection(2), Reverse(2),
               AffineCouplingBijection(cupNN()), ActNormBijection(2),
             ], context_init=uNN())


#---------------------------------# Training Controls #------------------------------------------#
# <editor-fold desc="Training Controls">
train = True
radiiNN = model.get_submodule('base_dist.radii_net')
radiiNN.train(mode=train)
radiiNN.requires_grad_(train)

train = True
osci_varNN = model.get_submodule('base_dist.osci_spread_net')
osci_varNN.train(mode=train)
osci_varNN.requires_grad_(train)

train = True
n_varNN = model.get_submodule('base_dist.n_var_net')
n_varNN.train(mode=train)
n_varNN.requires_grad_(train)

train = True
o_meanNN = model.get_submodule('base_dist.o_mean_net')
o_meanNN.train(mode=train)
o_meanNN.requires_grad_(train)

# <editor-fold desc="Secondary Training Controls">
uNN = model.get_submodule('context_init')
uNN.train(mode=True)
uNN.requires_grad_(True)

conc_cupc_NN = model.get_submodule('transforms.0')
conc_cupc_NN.train(mode=True)
conc_cupc_NN.requires_grad_(True)

cupNN1 = model.get_submodule('transforms.3')
cupNN1.train(mode=True)
cupNN1.requires_grad_(True)

cupNN2 = model.get_submodule('transforms.6')
cupNN2.train(mode=True)
cupNN2.requires_grad_(True)

cupNN3 = model.get_submodule('transforms.9')
cupNN3.train(mode=True)
cupNN3.requires_grad_(True)

misc1 = model.get_submodule('transforms.1')
misc1.train(mode=True)
misc1.requires_grad_(True)

misc2 = model.get_submodule('transforms.2')
misc2.train(mode=True)
misc2.requires_grad_(True)

misc4 = model.get_submodule('transforms.4')
misc4.train(mode=True)
misc4.requires_grad_(True)

misc5 = model.get_submodule('transforms.5')
misc5.train(mode=True)
misc5.requires_grad_(True)

misc7 = model.get_submodule('transforms.7')
misc7.train(mode=True)
misc7.requires_grad_(True)

misc8 = model.get_submodule('transforms.8')
misc8.train(mode=True)
misc8.requires_grad_(True)

misc10 = model.get_submodule('transforms.10')
misc10.train(mode=True)
misc10.requires_grad_(True)
# </editor-fold>
# </editor-fold>


#----------------------------------------# Optim #-----------------------------------------------#
optimizer = Adam(model.parameters(), lr=0.000075)


#========================================# Train #================================================#
saveFile = open('./data/osciCond_startTrain&Checkpoint.txt', 'w')

cond_colmn = 3
print("WARNING: cond_colmn STARTS AT INDEX=1")
total_epochs = 2000
for epoch in range(total_epochs):
    l = 0.0
    for i, datum in enumerate(train_loader):
        optimizer.zero_grad()
        (u, x) = mod_chooseDim.frontDim_split(datum, cond_colmn)
        loss = -model.log_prob(x, context=u).mean()

        # <editor-fold desc="Model State Print">
        #--------------------------------# Print ENTIRE Model State #------------------------------------#
        if(epoch in [1, 3499, 200]):
            print("Model's state parameters; epoch:"+str(epoch), file=saveFile)
            for p in model.state_dict():
                print(p, "\t", model.state_dict()[p].size(), model.state_dict()[p].data, file=saveFile)
        #-------------------------------------------------------------------------------------------------#
        # </editor-fold>

        loss.backward()
        optimizer.step()
        l += loss.detach().cpu().item()
        print('Epoch: {}/{}, Loglik: {:.3f}'.format(epoch+1, total_epochs, l/(i+1)), end='\r')
    print('')

saveFile.close()



#--------------------------------------# SAVE #-----------------------------------------------#
file = "./data/osciCond_MODELdata2.pt"
print("Saving to: ", file)
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, file)
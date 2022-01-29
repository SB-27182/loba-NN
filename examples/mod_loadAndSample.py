# Data:
import torch.nn as nn
import numpy as np

# Model:
import torch
from torch.distributions import Uniform
from survae.transforms import AffineCouplingBijection, ActNormBijection, Reverse
from survae.transforms import ConditionalAffineCouplingBijection
from survae.nn.layers import ElementwiseParams
# mod:
from survae.distributions.conditional._data_OsciCond_dist import dOscicond_Ovarmean_Nvar_radi_dist
from survae.flows._data_osciCond_flow import dOsciCond_flow

# Utils:
from survae.utils._new_conditioningFunctionOfVarbs import condVarb_partitioner
import random

# Plot:
import matplotlib.pyplot as plt

# Reproducibility:
torch.manual_seed(15)
np.random.seed(15)
random.seed(15)

cond_observations_count = 988
#================================# SubModule NN with dist #==================================#
class dist_net(nn.Module):
    def __init__(self, inDim, outDim):
        super(dist_net, self).__init__()
        self.lin = nn.Linear(inDim, outDim)

        #--------------------------------# Distribution & Sample #----------------------------------#
        dist = torch.distributions.Normal(0.0, 50.01)
        a = dist.sample([outDim, inDim]) #LEARNING EXPIRIENCE: FOR THE SAMPLE, IT'S ACTUALLY THE outDim THAT COMES FIRST (into columns, outto rows)-----> FOR SOME REASON YOU LIST INTO FIRST IN nn.Linear()
        a = a.detach()
        #-------------------------------------------------------------------------------------------#

        self.lin.weight = torch.nn.Parameter(a)
    def forward(self, x):
        return(self.lin(x))


#----------------------------# Model #-----------------------------------------#
# <editor-fold desc="OsciCond_nVar&oVar&radii Model">
def n_var_net():
    return nn.Sequential(dist_net(25, 200), nn.ReLU(),
                         nn.Linear(200, 100), nn.ReLU(),
                         nn.Linear(100, 100), nn.ReLU(),
                         nn.Linear(100, 50), nn.ReLU(),
                         nn.Linear(50, 2))

def cupNN():
    return nn.Sequential(nn.Linear(1, 200), nn.ReLU(), #The non-cond coupling layers must take in 1 because no cat with the context!
                         nn.Linear(200, 100), nn.ReLU(),
                         nn.Linear(100, 200), nn.ReLU(),
                         nn.Linear(200, 100), nn.ReLU(),
                         nn.Linear(100, 2), ElementwiseParams(2))

def cupcNN():
  return nn.Sequential(nn.Linear(2, 200), nn.ReLU(), #THIS HAS TO BE 2, THE cat(id, concOut) is what goes in!
                       nn.Linear(200, 100), nn.ReLU(),
                       nn.Linear(100, 200), nn.ReLU(),
                       nn.Linear(200, 100), nn.ReLU(),
                       nn.Linear(100, 2), ElementwiseParams(2)) #ALWAYS 2 BECAUSE WE ARE USING THE SCALE AND SHIFT IN SPACE TRANSFORM
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
# </editor-fold>



#---------------------------# Load #-------------------------------------------#
dictionary = torch.load("./data/osciCond_MODELdata2.pt")
model.load_state_dict(dictionary['model_state_dict'])
model.eval()


#-------------------# General Sample Generation#-------------------------------#
my_nVar_Dim = 0
latentColor = "green"
num_latent_sampls = 100
ll = -1
ul = 0
u = Uniform(ll, ul)
u_data = u.sample([cond_observations_count])
u_data = torch.reshape(u_data, (cond_observations_count, 1))
with torch.no_grad():
    samples_obs = model.sample(u_data).numpy()
    samples = model.sample_cond_latent_nVars(u_data , 2, myDim=my_nVar_Dim, stupidbatchsize=cond_observations_count, scaleMyVarBy=1.0, scaleOthrVarsBy=1.0)
    x_samples = samples[0].numpy()
    x_samples = x_samples[0:num_latent_sampls]
    z_samples = samples[1].numpy()
    z_samples = z_samples[0:num_latent_sampls]

# <editor-fold desc="#=== Plot Latent Samples Behavior ===#">
fig, ax = plt.subplots(1, 2, figsize=(12,6))
ax[0].set_title('Latent')
ax[0].scatter(z_samples[...,0], z_samples[...,1], s=20, facecolor="purple", edgecolor="white", alpha=0.9)
ax[0].set_xlim([-3,3])
ax[0].set_ylim([-3,3])
ax[1].set_title('U: LL='+ str(ll)+ ', UL='+ str(ul)+ "\nDIM: "+str(my_nVar_Dim))
ax[1].set_xlabel("X-AXIS")
ax[1].set_ylabel("Y-AXIS")
ax[1].scatter(samples_obs[...,0], samples_obs[...,1], s=20, alpha=0.2, facecolor="gray", edgecolor="gray")
ax[1].scatter(x_samples[...,0], x_samples[...,1], s=25, facecolor=latentColor, edgecolor="white", alpha=0.9)
ax[1].set_xlim([-10,10])
ax[1].set_ylim([-10,10])
plt.tight_layout()
plt.show()
# </editor-fold>


#-----------# Conditional Sliced Sample Generation and Plot #------------------#
ll = -1
ul = 4
partitions = 1
# <editor-fold desc="#===== Conditionally Sliced Sample Gen =====#">
parts = condVarb_partitioner(partitions, ll, ul)
with torch.no_grad():
    for i in parts:
        u = Uniform(i[0], i[1])
        u_partitioned_xData = u.sample([cond_observations_count])
        model_sample = model.sample(context=torch.reshape(u_partitioned_xData, (cond_observations_count, 1)))
        #dist = model.sample(u_partition)
        #sample = torch.squeez(dist.sample([1]))  #THIS is ommitted because the latent distribution is sampled with dist.rsample()
        # <editor-fold desc="Plotting Conditionaly Sliced">
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.set_title('U: LL='+ str(i[0].item())+ ', UL='+ str(i[1].item()))
        model_sample = model_sample.detach().numpy()
        ax.set_xlim([-10, 25])
        ax.set_xlabel("X-AXIS")
        ax.set_ylim([-10, 25])
        ax.set_ylabel("Y-AXIS")
        ax.scatter(model_sample[:, 0], model_sample[:, 1], s=20, facecolor="green", edgecolor="white", alpha=0.9)
        plt.show()
        # </editor-fold>
# </editor-fold>




#-----------# Ortho-Variance Mean Value As Function of u #------------------#
ll = 0
ul = 2.5
partitions = 40

parts = condVarb_partitioner(partitions, ll, ul)
var_line1 = []
var_line2 = []
x_ticks = []
with torch.no_grad():
    for i in parts:
        u = Uniform(i[0], i[1])
        x_ticks.append((i[0]+i[1])/2)
        u_partitioned_xData = u.sample([cond_observations_count])
        (var1, var2) = model.conditional_variance_avg(context=torch.reshape(u_partitioned_xData, (cond_observations_count, 1)))
        var_line1.append(var1)
        var_line2.append(var2)

fig, ax = plt.subplots(1, 1, figsize=(12, 12))
ax.plot(x_ticks, var_line1, color="purple")
ax.plot(x_ticks, var_line2, color="orange")
ax.legend(["Latent-Varb-1", "Latent-Varb-2"])
plt.ylabel("VARIANCE", fontsize=18)
plt.xlabel("CONDITIONING VARIABLE VALUE", fontsize=18)
plt.show()

# <editor-fold desc="Oscilator Vis. Currently Not Presenting.">
#-----------# Oscilator Analysis #------------------#
my_o_dim = 1
ll = -1
ul = 0
u = Uniform(ll, ul)
u_data = u.sample([cond_observations_count])
u_data = torch.reshape(u_data, (cond_observations_count, 1))
with torch.no_grad():
    samples_obs = model.sample(u_data).numpy()
    samples_o = model.latent_sample_1DimOscil(u_data, 2, myDim=my_o_dim, scaleMyVarBy=1.000, scaleMyOsciVarBy=1.00, scaleOthrVarsBy=0.10)

    samples_oscil = samples_o[0].numpy()

#fig, ax = plt.subplots(1, 1, figsize=(12,10))
#ax.set_title("Latent Transformed Oscilator Mapping\n"+'U: LL='+ str(ll)+ ', UL='+ str(ul)+ "\nDIM: "+str(my_o_dim))
#ax.scatter(samples_oscil[...,0], samples_oscil[...,1], s=20, facecolor="red", edgecolor="white", alpha=0.9)
#ax.scatter(samples_obs[...,0], samples_obs[...,1], s=20, alpha=0.2, facecolor="gray", edgecolor="gray")
#ax.set_xlim([-10,10])
#ax.set_ylim([-10,10])
#plt.show()
# </editor-fold>
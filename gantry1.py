
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
#import your_data_loading_module  # Replace with the actual module that loads your data
import new_sampler as samplers
import evaluatemodify as evaluate
import time, functools, torch, os,sys, random, fnmatch, psutil
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam,RAdam
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from prettytable import PrettyTable
import tqdm
from pickle import load
from IPython import display

sys.path.insert(1, '../')
#import trans_tdsm, utils
import data_utils as utils
import score_model as score_model
import sdes as sdes
#import util.display


device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:516"
os.system('nvidia-smi')

# Set padding value used
padding_value = 0.0

dataset = "dataset_2_padded_nentry"
preproc_dataset_name = 'firsttry'
dataset_store_path = os.path.join("/home/ken91021615/tdsm_encoder_new/datasets", preproc_dataset_name)
transform = None
transform_y = None
mask = True
jupyternotebook = True
workingdir = "./"

### SDE PARAMETERS ###
SDE = 'VE'
if SDE == 'VP':
    beta_max = 0.1
    beta_min = 0.005
if SDE == 'VE':
    sigma_max = 10.0
    sigma_min = 0.05
if SDE == 'subVP':
    beta_max = 1.0
    beta_min = 0.001 
    
### MODEL PARAMETERS ###
n_feat_dim = 4
embed_dim = 512
hidden_dim = 128
num_encoder_blocks = 8
num_attn_heads = 16
dropout_gen = 0
train_ratio = 0.8
batch_size = 128

# Instantiate stochastic differential equation
if SDE == 'VP':
    sde = sdes.VPSDE(beta_max=beta_max,beta_min=beta_min, device=device)
if SDE == 'VE':
    sde = sdes.VESDE(sigma_max=sigma_max,sigma_min=sigma_min,device=device)
if SDE == 'subVP':
    sde = sdes.subVPSDE(beta_max=beta_max,beta_min=beta_min, device=device)
marginal_prob_std_fn = functools.partial(sde.marginal_prob)
diffusion_coeff_fn = functools.partial(sde.sde)

print('torch version: ', torch.__version__)
print('Running on device: ', device)
if torch.cuda.is_available():
    print('Cuda used to build pyTorch: ',torch.version.cuda)
    print('Current device: ', torch.cuda.current_device())
    print('Cuda arch list: ', torch.cuda.get_arch_list())

print('Working directory: ', workingdir)

files_list_ = []
for filename in os.listdir(dataset_store_path):
    if fnmatch.fnmatch(filename, dataset + '*424To564.pt'):
        files_list_.append(os.path.join(dataset_store_path, filename))
print(files_list_)

# Instantiate model
model=score_model.Gen(n_feat_dim, embed_dim, hidden_dim, num_encoder_blocks, num_attn_heads, dropout_gen, marginal_prob_std=marginal_prob_std_fn)
torch.save(model.state_dict(), 'initial_model.pt')

# Initialize your Classifier and Generator
embed_dim=64
hidden_dim=64
n_layers=2
n_layers_cls=2
n_heads=2
dropout=0
classifier = evaluate.Classifier(4, embed_dim, hidden_dim, n_layers, n_layers_cls, n_heads, dropout)
generator = samplers.pc_sampler(sde, padding_value, snr=0.2, sampler_steps=100, steps2plot=(), device='cuda', eps=1e-3, jupyternotebook=False)
criterion = torch.nn.BCELoss()
# Initialize optimizers for both classifier and generator
optimizer_cls = optim.Adam(classifier.parameters(), lr=1e-4)
optimizer_gen = optim.Adam(generator.parameters(), lr=1e-4)

# Set up DataLoader for your training data
# Replace 'YourDataset' and 'your_data_loading_function' with your actual dataset and loading function
##train_dataset = YourDataset(...)
#train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Set the number of training epochs
num_epochs = 150

# Training loop
lrs_ = []

print(files_list_)
eps_ = []
for epoch in range(num_epochs):
    eps_.append(epoch)
    # Create/clear per epoch variables
    cumulative_epoch_loss = 0.
    cumulative_test_epoch_loss = 0.

    file_counter = 0
    n_training_showers = 0
    n_testing_showers = 0
    training_batches_per_epoch = 0
    testing_batches_per_epoch = 0
    n_showers_2_gen = 10000
    # Load files
    idx=0
    for filename in files_list_:
        idx+=1
        custom_data = utils.cloud_dataset(filename, device=device)
        train_size = int(train_ratio * len(custom_data.data))
        test_size = len(custom_data.data) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(custom_data, [train_size, test_size])
        n_training_showers+=train_size
        n_testing_showers+=test_size
        
        # Load clouds for each epoch of data dataloaders length will be the number of batches
        shower_loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        shower_loader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        # Accumuate number of batches per epoch
        training_batches_per_epoch += len(shower_loader_train)
        testing_batches_per_epoch += len(shower_loader_test)
        n_files = len(files_list_)
        nshowers_per_file = [n_showers_2_gen//n_files for x in range(n_files)]

        # Load shower batch for training
        for i, (shower_data, incident_energies) in enumerate(shower_loader_train):
            classifier.train()
            optimizer_cls.zero_grad()

            # Forward pass through the discriminator
            real_labels = torch.ones(shower_data.size(0), device=device)
            fake_labels = torch.zeros(shower_data.size(0), device=device)

            real_output = classifier(shower_data, incident_energies)
            # Train the discriminator (Classifier)
            sample_ine=[]
            sample_ = []
            valid_event = []
            data_np = shower_data.cpu().numpy().copy()
            energy_np = incident_energies.cpu().numpy().copy()

            
            # Mask for padded values
            masking = data_np[:,:,0] != padding_value 
            
            # Loop over each shower in batch
            for j in range(len(data_np)):
                
                # valid hits for shower j in batch used for GEANT plot distributions
                valid_hits = data_np[j]
                
                # real (unpadded) hit multiplicity needed for the 2D PDF later
                n_valid_hits = data_np[j][masking[j]]
                
                if len(valid_hits)>max_hits:
                    max_hits = len(valid_hits)

                n_valid_hits_per_shower = np.append(n_valid_hits_per_shower, len(n_valid_hits))
                incident_e_per_shower = np.append(incident_e_per_shower, energy_np[j])
            in_energies = torch.from_numpy(np.random.choice( incident_e_per_shower, nshowers_per_file[idx] ))
            if idx == 0:
                sampled_ine = in_energies
            else:
                sampled_ine = torch.cat([sampled_ine,in_energies])
            
            # Create variable length tensors of random noise for features of hits
            e_vs_nhits_prob, x_bin, y_bin = samplers.get_prob_dist(incident_e_per_shower, n_valid_hits_per_shower, 20)
            nhits, gen_hits = samplers.generate_hits(e_vs_nhits_prob, x_bin, y_bin, in_energies, 4, device=device)
            
            # Save
            torch.save([gen_hits, in_energies],'tmp.pt')

            # Load the showers of noise
            gen_hits = utils.cloud_dataset('tmp.pt', device=device)
            
            # Match padding length of training files
            gen_hits.max_nhits = max_hits

            # Pad showers to have equal length
            gen_hits.padding(value=padding_value)
            
            # Load len(gen_hits_loader) number of batches each with batch_size number of showers
            gen_hits_loader = DataLoader(gen_hits, batch_size=batch_size, shuffle=False)

            # Remove noise shower file
            os.system("rm tmp.pt")

            # Load a batch of noise showers
            sample = []
            for i, (gen_hit, sampled_energies) in enumerate(gen_hits_loader,0):
                print(f'Batch: {i}')
                print(f'Generation batch {i}: showers per batch: {gen_hit.shape[0]}, max. hits per shower: {gen_hit.shape[1]}, features per hit: {gen_hit.shape[2]}, sampled_energies: {len(sampled_energies)}')    
                sys.stdout.write('\r')
                sys.stdout.write("Progress: %d/%d" % ((i+1), len(gen_hits_loader)))
                sys.stdout.flush()

                # Run reverse diffusion sampler on showers of random noise + zero-padded hits
                #generative = sampler(model, marginal_prob_std_fn, diffusion_coeff_fn, sampled_energies, gen_hit, batch_size=gen_hit.shape[0])
                generative = generator(model, sampled_energies, gen_hit, batch_size=gen_hit.shape[0])
                #fake_output = classifier(generator(score_model, sampled_energies, gen_hit, batch_size), incident_energies)

                # Create first sample or concatenate sample to sample list
                if i == 0:
                    sample = generative
                else:
                    sample = torch.cat([sample,generative])

                #print(f'sample: {sample.shape}')

                sample_np = sample.cpu().numpy()

                for i in range(len(sample_np)):
                # The output of the sampler has nhits == max hits (do we want to remove these here or during the diffusion?)
                    tmp_sample = sample_np[i]#[:nhits[i]]
                    sample_.append(torch.tensor(tmp_sample))

    fake_output = classifier(sample_,sample_ine)
            

        

        
    #fake_output move to generative part
            

    # Compute discriminator loss
    cls_loss_real = criterion(real_output, real_labels)
    cls_loss_fake = criterion(fake_output, fake_labels)
    cls_loss = cls_loss_real + cls_loss_fake

            # Backward and optimize
    cls_loss.backward()
    optimizer_cls.step()

            # Train the generator
    generator.eval()  # Set the generator to evaluation mode
    optimizer_gen.zero_grad()

            # Generate fake data
    generated_data = [sample_,sample_ine]

                # Forward pass through the discriminator with generated data
    discriminator_output = classifier(generated_data, incident_energies)

                # Compute generator loss
    gen_loss = criterion(discriminator_output, real_labels)

                # Backward and optimize
    gen_loss.backward()
    optimizer_gen.step()
    
        # Print some statistics or visualize the training progress if needed

    # Evaluate the model or generate samples after each epoch if needed
    # ...

# Save or use the trained models for further tasks
torch.save(classifier.state_dict(), 'classifier.pth')
torch.save(generator.state_dict(), 'generator.pth')

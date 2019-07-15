import torch
import torchvision

from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

import imageio
from torchvision.utils import make_grid

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 64
img_transform = transforms.Compose([transforms.ToTensor()])
train_dataset = MNIST(root='./MNIST_data', train=True, transform=img_transform, download=True)
test_dataset = MNIST(root='./MNIST_data', train=False, transform=img_transform, download=True)

train_data = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True)
test_data = DataLoader(test_dataset, batch_size=10000, num_workers=4, pin_memory=True, shuffle=True)

input_size = 28 * 28
EPS = 1e-12

class JointVAE(nn.Module):
    def __init__(self, latent_cont_dim, latent_disc_dim, temperature=.67):
        super(JointVAE, self).__init__()

        self.hidden_dim = 64
        self.latent_cont_dim = latent_cont_dim
        self.latent_disc_dim = latent_disc_dim
        self.temperature = temperature
        self.reshape = (12, 3, 3)
        self.num_pixels = input_size
        self.training = False

        self.conv_encode = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=4, stride=2, padding=1), nn.ReLU(), #nn.BatchNorm2d(14*14),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=1), nn.ReLU(), #nn.BatchNorm2d(7*7),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # Final dimension = batch_size x 12 x 3 x 3
        )
        self.fully_connected_encode = nn.Sequential(
            nn.Linear(12*3*3, self.hidden_dim), nn.ReLU()
        )

        self.mean = nn.Linear(self.hidden_dim, self.latent_cont_dim)
        self.log_var = nn.Linear(self.hidden_dim, self.latent_cont_dim)
        self.mass = nn.Linear(self.hidden_dim, self.latent_disc_dim)


        self.fully_connected_decode = nn.Sequential(
            nn.Linear(self.latent_cont_dim + self.latent_disc_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, 12*3*3), nn.ReLU()
        )
        self.conv_decode = nn.Sequential(
            nn.ConvTranspose2d(in_channels=12, out_channels=6, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(in_channels=6, out_channels=6, kernel_size=3, stride=2, padding=0, dilation=2), nn.ReLU(),
            nn.ConvTranspose2d(in_channels=6, out_channels=1, kernel_size=4, stride=2, padding=0, dilation=3, output_padding=2), nn.Sigmoid()
        )

    def encode(self, x):
        # Encodes an image into parameters of a latent distribution
        batch_size = x.size()[0]

        temp = self.fully_connected_encode(self.conv_encode(x).view(-1, 12*3*3))
        mu = self.mean(temp)
        log_sigma = self.log_var(temp)
        alpha = nn.Softmax(dim=1)(self.mass(temp))

        return mu, log_sigma, alpha

    def reparametrize(self, latent_params):
        if self.training:
            # Sample continuous latent variable
            std = torch.exp(0.5 * latent_params[1])
            eps = torch.randn(std.size()).to(device)
            cont_sample = latent_params[0] + std * eps

            # Sample discrete latent variable
            disc_sample = self.gumble_softmax(latent_params[2])

            return torch.cat((cont_sample, disc_sample), dim=1)
        else:
            # When in Reconstruction, simply set continuous latent variable to be the mean
            cont_sample = latent_params[0]

            # When in Reconstruction, simply pick the most probable one
            _, max_alpha_idx = torch.max(latent_params[2], dim=1)
            one_hot_samples = torch.zeros(latent_params[2].size())
            one_hot_samples.scatter_(1, max_alpha_idx.view(-1, 1).data.cpu(), 1)
            return torch.cat((cont_sample, one_hot_samples), dim=1)

    def gumble_softmax(self, alpha):
        unif = torch.rand(alpha.size()).to(device)
        gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
        log_alpha = torch.log(alpha + EPS)
        logit = (log_alpha + gumbel) / self.temperature

        return nn.functional.softmax(logit, dim=1)

    def decode(self, latent_sample):
        features = self.fully_connected_decode(latent_sample)
        return self.conv_decode(features.view(-1, *self.reshape))

    def forward(self, x):
        latent_params = self.encode(x)
        latent_sample = self.reparametrize(latent_params)
        return self.decode(latent_sample), latent_params, latent_sample



class Train:
    def __init__(self, model, optimizer, cont_capacity=None, disc_capacity=None, print_loss_every=500, record_loss_every=5):
        self.model = model                      # JointVAE
        self.optimizer = optimizer              # torch.optim.Optimizer instance
        self.cont_capacity = cont_capacity      # tuple (min_capacity, max_capacity, num_iters, gamma_z)
        self.disc_capacity = disc_capacity      # tuple (min_capacity, max_capacity, num_iters, gamma_c)
        self.print_loss_every = print_loss_every
        self.record_loss_every = record_loss_every

        self.num_steps = 0
        self.batch_size = None
        self.losses = {'loss':[], 'recon_loss':[], 'kl_loss':[]}

        self.losses['kl_loss_cont'] = []
        for i in range(self.model.latent_cont_dim):
            self.losses['kl_loss_cont_' + str(i)] = []

        for i in range(1):
            self.losses['kl_loss_disc_' + str(i)] = []

    def train(self, data_loader, epochs=1, save_training_gif=None):
        if save_training_gif is not None:
            training_progress_images = []

        self.batch_size = data_loader.batch_size
        self.model.training = True

        for epoch in range(epochs):
            mean_epoch_loss = self._train_epoch(data_loader)
            print('Epoch: {} Average Loss: {:.2f}'.format(epoch+1, self.batch_size * self.model.num_pixels * mean_epoch_loss))

            if save_training_gif is not None:
                # Generate batch of images and convert to grid
                viz = save_training_gif[1]
                viz.save_images = False
                img_grid = viz.all_latent_traversals(size=10)
                # Convert to numpy and transpose axes to fit imageio convention
                # i.e. (width, height, channels)
                img_grid = np.transpose(img_grid.numpy(), (1,2,0))
                # Add image grid to training progress
                training_progress_images.append(img_grid)

        if save_training_gif  is not None:
            imageio.mimsave(save_training_gif[0], training_progress_images, fps=24)

    def _train_epoch(self, data_loader):
        epoch_loss = 0
        print_every_loss = 0

        for batch_idx, (data, label) in enumerate(data_loader):
            iter_loss = self._train_iteration(data)
            epoch_loss += iter_loss
            print_every_loss += iter_loss

            if batch_idx % self.print_loss_every == 0:
                if batch_idx == 0:
                    mean_loss = print_every_loss
                else:
                    mean_loss = print_every_loss / self.print_loss_every
                print('{}/{}\tLoss: {:3f}'.format(batch_idx * len(data), len(data_loader.dataset), self.model.num_pixels * mean_loss))

                print_every_loss = 0
        # Mean Epoch Loss
        return epoch_loss / len(data_loader.dataset)

    def _train_iteration(self, data):
        self.num_steps += 1
        data = data.to(device)

        self.optimizer.zero_grad()
        recon_batch, latent_params, _ = self.model(data)
        loss = self.loss_function(data, recon_batch, latent_params)
        loss.backward()
        self.optimizer.step()

        train_loss = loss.item()
        return train_loss

    def loss_function(self, data, recon_data, latent_params):
        # data shape (N, C, H, W)
        # latent_params shape (mu, log_sigma^2, alpha)

        Recon_loss = torch.sum((recon_data.view(-1, self.model.num_pixels) - data.view(-1, self.model.num_pixels)) ** 2)
        #Recon_loss = - torch.sum(data.view(-1, self.model.num_pixels) * torch.log(recon_data.view(-1, self.model.num_pixels)) +(1-data.view(-1, self.model.num_pixels)) * torch.log(1-recon_data.view(-1, self.model.num_pixels)), dim=1)
        #Recon_loss = nn.functional.binary_cross_entropy(recon_data.view(-1, self.model.num_pixels), data.view(-1, self.model.num_pixels))
        #Recon_loss *= self.model.num_pixels

        # KL divergences
        kl_cont_loss = 0
        kl_disc_loss = 0
        cont_capacity_loss = 0
        disc_capacity_loss = 0

        '''For Continuous Variables'''
        mean, log_var = latent_params[0:2]
        kl_cont_loss =self._kl_normal_loss(mean, log_var)
        # Linearly increase capacity of continuous channels
        cont_min, cont_max, cont_num_iters, cont_gamma = self.cont_capacity

        cont_cap_current = (cont_max - cont_min) * self.num_steps / float(cont_num_iters) + cont_min
        cont_cap_current = min(cont_cap_current, cont_max)

        cont_capacity_loss = cont_gamma * torch.abs(cont_cap_current - kl_cont_loss)

        '''For Discrete Variables'''
        kl_disc_loss = self._kl_discrete_loss(latent_params[2])

        disc_min, disc_max, disc_num_iters, disc_gamma = self.disc_capacity

        disc_cap_current = (disc_max - disc_min) * self.num_steps / float(disc_num_iters) + disc_min
        disc_cap_current = min(disc_cap_current, disc_max)
        # Require float conversion here to not end up with numpy float
        disc_theoretical_max = float(np.log(self.model.latent_disc_dim))
        disc_cap_current = min(disc_cap_current, disc_theoretical_max)

        disc_capacity_loss = disc_gamma * torch.abs(disc_cap_current - kl_disc_loss)

        # Calculate total kl value to record it
        kl_loss = kl_cont_loss + kl_disc_loss

        # Calculate total loss
        total_loss = Recon_loss + cont_capacity_loss + disc_capacity_loss

        if self.model.training and self.num_steps % self.record_loss_every == 1:
            self.losses['recon_loss'].append(Recon_loss.item())
            self.losses['kl_loss'].append(kl_loss.item())
            self.losses['loss'].append(total_loss.item())

        return total_loss / self.model.num_pixels

    def _kl_normal_loss(self, mean, logvar):
        kl_values = -0.5 * (1+logvar - mean.pow(2) - logvar.exp())
        kl_means = torch.mean(kl_values, dim=0)
        kl_loss = torch.sum(kl_means)

        if self.model.training and self.num_steps % self.record_loss_every == 1:

            self.losses['kl_loss_cont'].append(kl_loss.item())
            for i in range(self.model.latent_cont_dim):
                self.losses['kl_loss_cont_'+str(i)].append(kl_means[i].item())

        return kl_loss

    def _kl_discrete_loss(self, alpha):
        disc_dim = alpha.size()[-1] # Should be type int
        log_dim = torch.Tensor([np.log(self.model.latent_disc_dim)]).to(device)

        neg_entropy = torch.sum(alpha * torch.log(alpha + EPS), dim=1)

        # Mean over the batch size
        mean_neg_entropy = torch.mean(neg_entropy, dim=0)

        #KL loss of alpha with uniform categorical variable
        kl_loss = log_dim + mean_neg_entropy

        return kl_loss

batch_size = 64
lr = 5e-4
epochs = 50

data_loader, _ = train_data, test_data
img_size = (1, 28, 28)

model = JointVAE(100, 10, .67).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
trainer = Train(model, optimizer, cont_capacity=[0.0, 5.0, 25000, 30],
                                  disc_capacity=[0.0, 5.0, 25000, 30])

trainer.train(data_loader, epochs)

model.training = False
for data in test_data:
    input_test, label_test = data
    output_test, latent_params, latent_vars = model(input_test)

from sklearn.decomposition import PCA
PCA = PCA(n_components=2)
reduced_array = latent_vars[:, :20].detach().numpy()
reduced_array = PCA.fit_transform(reduced_array)

plt.figure(figsize=(7,7))
plt.scatter(reduced_array[:,0], reduced_array[:,1], c=label_test, cmap='nipy_spectral', s=4)


original = input_test[1]
original_label = label_test[1]

plt.figure(figsize=(5,5))
plt.imshow(original.detach().numpy()[0], cmap='gray')
plt.imshow(output_test[1].detach().numpy()[0], cmap='gray')


import random
idx = random.choice(list(i for i in range(10000)))
plt.figure(figsize=(5,5))
plt.imshow(output_test[idx].detach().numpy()[0], cmap='gray')

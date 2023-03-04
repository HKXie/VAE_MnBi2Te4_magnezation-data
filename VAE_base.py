import torch
import torch.nn as nn
import numpy as np

# functions to generate random data

def generate_random_image(batch_size,size):
    random_data = torch.rand(batch_size,size)#0-1
    return random_data


def generate_random_seed(batch_size,size):
    random_data = torch.randn(batch_size,size)#Standard normal distribution
    return random_data

def generate_random_one_hot(batch_size,size):
    label_tensor=torch.zeros(batch_size,size)
    for i in range(batch_size):
        randon_idx=np.random.randint(0,size)
        label_tensor[i, randon_idx]=1.0
    return label_tensor


#VAE，Encoder and Decoder
class VAE(nn.Module):
    def __init__(self, image_size=3*10*10, h_dim=400, z_dim=20,learning_rate=1e-4, class_num=50):
        super().__init__()
        self.fc1 = nn.Linear(image_size+class_num, h_dim)

        self.fc2 = nn.Linear(h_dim, z_dim)#for mu
        self.fc3 = nn.Linear(h_dim, z_dim)#for var

        self.fc4 = nn.Linear(z_dim+class_num, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)

        # self.act1 = nn.ReLU()
        # self.act1 = nn.LeakyReLU(0.02)
        
        self.act1 = nn.GELU()
        self.act2 = nn.Sigmoid()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # self.train_Loss=0
        # self.kl_Loss=0
        # self.reconstruction_Loss=0
 
    def encode(self, x, label_tensor):
        x = torch.cat((x,label_tensor),axis=1)
        output = self.fc1(x)
        output = self.act1(output)

        output_mu = self.fc2(output)
        output_var = self.fc3(output)

        return output_mu, output_var
 
    def reparameterize(self, mu, log_var):

        var = torch.exp(log_var)#/2
        eps = torch.randn_like(var)

        return mu + eps * var
 
    def decode(self, z, label_tensor):
        # h = F.relu(self.fc4(z))
        z = torch.cat((z,label_tensor), axis=1)
        output = self.fc4(z)
        output = self.act1(output)
        output = self.fc5(output)
        output = self.act2(output)
        return output
 
    def forward(self, x, label_tensor):
        mu, log_var = self.encode(x, label_tensor)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z, label_tensor)
        return x_reconst, mu, log_var
 
 
 
    # def loss_function(self, x_reconst, x, mu, log_var, batch_size): #损失函数，reconstructed loss + KL loss 

    #     Loss = nn.BCELoss(reduction='sum')#
    #     # Loss = nn.MSELoss(reduction='sum')#

    #     # Loss = nn.CrossEntropyLoss(reduction='sum')

    #     reconstruction_loss = Loss(x_reconst, x)/batch_size

    #     KL_divergence = -0.5 * torch.sum(1 + log_var - torch.exp(log_var) - mu ** 2)/batch_size
        
    #     return reconstruction_loss + KL_divergence, reconstruction_loss, KL_divergence


    # def Train(self, gen_imgs, real_imgs, mu, log_var, batch_size): 
        
    #     train_loss, reconstruction_loss, kl_loss = self.loss_function(gen_imgs, real_imgs, mu, log_var, batch_size)

    #     self.train_Loss = train_loss#用于记录loss,kl+recons       
    #     self.reconstruction_Loss = reconstruction_loss
    #     self.kl_Loss = kl_loss#

    #     # back propagation
    #     self.optimizer.zero_grad()
    #     train_loss.backward()
    #     self.optimizer.step()

        # return loss

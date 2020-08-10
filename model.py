import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh()
        ) 

    def forward(self, z):
        z = z.view(z.size(0), 100)
        ret = self.model(z)
        return ret.view(z.size(0), -1, 28, 28)
         
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), 784)
        ret = self.model(x)
        return ret

if __name__ == '__main__':
    batch_size = 1

    generator = Generator()
    generator.cuda()
    noise = torch.rand(batch_size, 100).cuda()
    gen = generator(noise)
    print(gen.shape)

    discriminator = Discriminator()
    discriminator.cuda()
    image = torch.rand(batch_size, 1, 28, 28).cuda()
    dis = discriminator(image)
    print(dis.shape)
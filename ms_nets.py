import torch
import torch.nn as nn
import torch.nn.functional as F


class Net4Conv(nn.Module):
    def __init__(self):
        super(Net4Conv, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.conv3 = nn.Conv2d(50, 50, 3, 1, 1)
        self.conv4 = nn.Conv2d(50, 50, 3, 1, 1)
        self.fc1 = nn.Linear(50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Net3Conv(nn.Module):
    def __init__(self):
        super(Net3Conv, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.conv3 = nn.Conv2d(50, 50, 3, 1, 1)
        self.fc1 = nn.Linear(2*2*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 2*2*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class NetGenBase(nn.Module):

    def __init__(self, z_dim):
        super(NetGenBase, self).__init__()
        self.z_dim = z_dim
        self.decoder = None
        self.fc_dinput = None

    def decode(self, z) -> torch.Tensor:
        pass

    def forward(self, z, no_act=False):
        x = self.decode(z)
        if not no_act:
            x = torch.relu(x)
        return x

    def sample(self, num_samples):
        z = torch.randn(num_samples, self.z_dim)
        samples = self.decode(z)
        return samples

    def generate(self, z):
        return self.decode(z)


class NetGenMnist(NetGenBase):

    def __init__(self, z_dim=128):
        super(NetGenMnist, self).__init__(z_dim)

        dim = 5 * 4 * 4
        self.fc_dinput = nn.Linear(self.z_dim, dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(5, 10, 5, stride=1),  # 5*4*4=>10*8*8
            nn.BatchNorm2d(10),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(10, 10, 5, stride=4),  # 10*8*8=>10*33*33
            nn.BatchNorm2d(10),
            nn.LeakyReLU(),
            nn.Conv2d(10, 1, 6, stride=1),  # 10*33*33=>1*28*28
            nn.BatchNorm2d(1),
        )

    def decode(self, z) -> torch.Tensor:
        x = self.fc_dinput(z)
        x = x.view(x.shape[0], 5, 4, 4)
        # print("x:", x.shape)
        x = self.decoder(x)
        return x


class NetGen(nn.Module):
    def __init__(self, nz=128, nc=1, img_size=28, ngf=64):
        super(NetGen, self).__init__()

        # self.activation = torch.tanh
        self.activation = torch.relu

        self.init_size = img_size//4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf*2*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(ngf*2),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.BatchNorm2d(nc, affine=False)
        )

    def forward(self, z, no_act=False):
        out = self.l1(z.view(z.shape[0], -1))
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)

        if no_act:
            return img
        else:
            return self.activation(img)


class NetGenSMnist(nn.Module):

    def __init__(self, z_dim=128):
        super(NetGenSMnist, self).__init__()

        self.z_dim = z_dim

        dim = 5 * 4 * 4
        self.fc_dinput = nn.Linear(self.z_dim, dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(5, 10, 5, stride=1),  # 5*4*4=>10*8*8
            nn.BatchNorm2d(10),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(10, 10, 5, stride=4),  # 10*8*8=>10*33*33
            nn.BatchNorm2d(10),
            nn.LeakyReLU(),
            # nn.Conv2d(10, 1, 6, stride=1),  # 10*33*33=>1*28*28
            # nn.BatchNorm2d(1),
        )

    def decode(self, z) -> torch.Tensor:
        x = self.fc_dinput(z)
        x = x.view(x.shape[0], 5, 4, 4)
        # print("x:", x.shape)
        x = self.decoder(x)
        return x

    def forward(self, z, no_act=False):
        x = self.decode(z)
        return x


class NetGenEMnist(nn.Module):

    def __init__(self):
        super(NetGenEMnist, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(10, 1, 6, stride=1),  # 10*33*33=>1*28*28
            nn.BatchNorm2d(1),
        )

    def forward(self, x, no_act=False):
        x = self.layers(x)

        if not no_act:
            x = torch.relu(x)
        return x


class NetGenS(nn.Module):
    def __init__(self, nz=128, nc=1, img_size=28, ngf=64):
        super(NetGenS, self).__init__()

        self.init_size = img_size//4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf*2*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(ngf*2),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            # nn.BatchNorm2d(nc, affine=False)
        )

    def forward(self, z, no_act=False):
        out = self.l1(z.view(z.shape[0], -1))
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)

        return img


class NetGenE(nn.Module):

    def __init__(self, nz=128, nc=1, img_size=28, ngf=64, act=None):
        super(NetGenE, self).__init__()

        if act is None:
            self.activation = torch.relu
        else:
            self.activation = act

        self.layers = nn.Sequential(
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.BatchNorm2d(nc, affine=False)
        )

    def forward(self, x, no_act=False):
        img = self.layers(x)

        if no_act:
            return img
        else:
            return self.activation(img)

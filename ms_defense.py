import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from ms_nets import Net3Conv, Net4Conv
from resnet import ResNet50, ResNet18, ResNet34
from models.resnet_cifar100 import resnet34
from models.vgg import VGG16OC

import ms_common as comm


class MSDefense(object):

    def __init__(self, args):
        super(MSDefense, self).__init__()
        self.args = args

        self.netV = None

        self.test_loader = None
        self.train_loader = None

    def load(self, netv_path=None):
        """
        Loading nets and datasets
        """
        if self.args.dataset == 'MNIST' or self.args.dataset == 'FashionMNIST':
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            if self.args.dataset == 'MNIST':
                test_set = torchvision.datasets.MNIST(root='dataset/', train=False, download=True, transform=transform)
                train_set = torchvision.datasets.MNIST(root='dataset/', train=True, download=True, transform=transform)
            if self.args.dataset == 'FashionMNIST':
                test_set = torchvision.datasets.FashionMNIST(root='dataset/', train=False, download=True, transform=transform)
                train_set = torchvision.datasets.FashionMNIST(root='dataset/', train=True, download=True, transform=transform)

            self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=500, num_workers=2)
            self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=50, shuffle=True, num_workers=2)
        if self.args.dataset == 'SVHN':
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            test_set = torchvision.datasets.SVHN(root='dataset/', split='test', download=True, transform=transform)
            train_set = torchvision.datasets.SVHN(root='dataset/', split='train', download=True, transform=transform)

            self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)
            self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
        if self.args.dataset == 'CIFAR10':
            transform_train = torchvision.transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            transform_test = torchvision.transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform_test)
            train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform_train)

            self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)
            self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
        if self.args.dataset == 'CIFAR100':
            transform_train = torchvision.transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                # transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
            ])
            transform_test = torchvision.transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),(0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
            ])
            test_set = torchvision.datasets.CIFAR100(root='./dataset', train=False, download=True, transform=transform_test)
            train_set = torchvision.datasets.CIFAR100(root='./dataset', train=True, download=True, transform=transform_train)

            self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)
            self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
        self.set_netV(netv_path=netv_path)

    def set_netV(self, netv_path=None):
        if self.args.dataset == 'MNIST':
            net_name = 'VGG16OC'
        elif self.args.dataset == 'FashionMNIST':
            net_name = 'VGG16OC'
        elif self.args.dataset == 'CIFAR100':
            net_name = 'ResNet34_100'
        else:  # cifar10, SVHN
            net_name = 'ResNet34'

        if net_name == 'Net4Conv':
            VNet = Net4Conv
        elif net_name == 'VGG16OC':
            VNet = VGG16OC
        elif net_name == 'ResNet34':
            VNet = ResNet34
        elif net_name == 'ResNet34_100':
            def net():
                return ResNet34(num_classes=100)
            VNet = net
        else:
            VNet = None

        if self.args.cuda:
            self.netV = VNet().cuda()
            map_location = lambda storage, loc: storage.cuda()
        else:
            self.netV = VNet().cpu()
            map_location = 'cpu'
        self.netV = nn.DataParallel(self.netV)
        if netv_path is not None:
            state_dict = torch.load(netv_path, map_location=map_location)
            self.netV.load_state_dict(state_dict)
        self.netV.eval()
        print("The architecture of the target model:", net_name)

    def train_netV(self, save_path):
        print("Starting training net V")
        t_net = resnet34().cuda()
        state_dict = torch.load('saved_model/res/resnet34-194-best.pth')
        t_net.load_state_dict(state_dict)
        t_net.eval()
        acc = comm.accuracy(t_net, 't_net', test_loader=self.test_loader, cuda=self.args.cuda)
        import torchvision.transforms.functional as Ft

        # optimizer = torch.optim.Adam(self.netV.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        if self.args.dataset == 'CIFAR10' or self.args.dataset == 'CIFAR100':
            # optimizer = torch.optim.SGD(self.netV.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
            optimizer = torch.optim.RMSprop(self.netV.parameters(), lr=0.0001)
        else:
            optimizer = torch.optim.RMSprop(self.netV.parameters(), lr=0.0001)

        self.netV.train()
        criterion = nn.CrossEntropyLoss()

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 160, 180], gamma=0.2)

        best_acc = 0
        for epoch in range(self.args.epoch_b):
            print("epoch: %d / %d" % (epoch+1, self.args.epoch_b))

            for idx, data in enumerate(self.train_loader, 0):
                inputs, labels = data

                if self.args.cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                else:
                    inputs = inputs.cpu()
                    labels = labels.cpu()

                optimizer.zero_grad()

                outputs = self.netV(inputs)
                # loss = criterion(outputs, labels)

                # inputs_norm = Ft.normalize(inputs, (0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404), False)
                t_outputs = t_net(inputs)
                loss = F.l1_loss(outputs, t_outputs)

                if idx % 100 == 0:
                    print("loss:", loss)

                loss.backward()

                optimizer.step()

            acc = comm.accuracy(self.netV, 'netV', test_loader=self.test_loader, cuda=self.args.cuda)
            if acc > best_acc:
                torch.save(self.netV.state_dict(), save_path+str(epoch)+".pth")
                best_acc = acc
            # if self.args.dataset == 'CIFAR10':
            #     scheduler.step()
            scheduler.step()

        # torch.save(self.netV.state_dict(), save_path)
        print("Finished training of netV")




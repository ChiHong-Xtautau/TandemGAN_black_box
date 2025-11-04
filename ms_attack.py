import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
# from advertorch.attacks import LinfBasicIterativeAttack, GradientSignAttack
# from advertorch.attacks import CarliniWagnerL2Attack, PGDAttack

from ms_nets import Net3Conv, Net4Conv
from ms_nets import NetGenMnist, NetGen
from models.resnet import ResNet18
from models.vgg import VGG11, VGG11OC
from models.alexnet import AlexNet
from models.lenet import LeNetMnist

import ms_common as comm


class MSAttack(object):

    def __init__(self, args, defense_obj=None):
        self.args = args
        self.device = torch.device("cuda:0" if self.args.cuda else "cpu")

        self.netS = None

        self.netG = None
        self.z_dim = args.z_dim

        self.msd = defense_obj
        self.adversary = None

        self.train_loader = None
        self.test_loader = None

    def load(self, nets_path=None, net_type="Net3Conv"):
        """
        Loading nets and datasets
        """
        if self.args.dataset == 'MNIST':
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            test_set = torchvision.datasets.MNIST(root='dataset/', train=False, download=True, transform=transform)
            train_set = torchvision.datasets.MNIST(root='dataset/', train=True, download=True, transform=transform)
        if self.args.dataset == 'FashionMNIST':
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            test_set = torchvision.datasets.FashionMNIST(root='dataset/', train=False, download=True, transform=transform)
            train_set = torchvision.datasets.FashionMNIST(root='dataset/', train=True, download=True, transform=transform)
        if self.args.dataset == 'SVHN':
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            test_set = torchvision.datasets.SVHN(root='dataset/', split='test', download=True, transform=transform)
            train_set = torchvision.datasets.SVHN(root='dataset/', split='train', download=True, transform=transform)
        if self.args.dataset == "CIFAR10":
            transform_train = torchvision.transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            transform_test = torchvision.transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform_test)
            train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform_train)

        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=500, shuffle=False, num_workers=2)
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=50, shuffle=True, num_workers=2)
        self.set_netS(nets_path=nets_path)

        self.test_loader_supp = torch.utils.data.DataLoader(test_set, batch_size=30, shuffle=True, num_workers=2)

    def set_netS(self, nets_path=None):
        if self.args.dataset == 'MNIST':
            s_net_name = 'VGG11OC'
        elif self.args.dataset == 'FashionMNIST':
            s_net_name = 'VGG11OC'
        else:
            if self.args.l_only:
                s_net_name = 'VGG11'
            else:
                s_net_name = 'VGG11'

        if s_net_name == 'LeNetMnist':
            SNet = LeNetMnist
        elif s_net_name == 'ResNet18':
            SNet = ResNet18
        elif s_net_name == 'VGG11':
            SNet = VGG11
        elif s_net_name == 'AlexNet':
            SNet = AlexNet
        elif s_net_name == 'VGG11OC':
            SNet = VGG11OC
        elif s_net_name == 'Net3Conv':
            SNet = Net3Conv
        else:
            SNet = None

        self.netS = SNet().to(self.device)

        self.netS = nn.DataParallel(self.netS)
        if self.args.cuda:
            map_location = lambda storage, loc: storage.cuda()
        else:
            map_location = 'cpu'
        if nets_path is not None:
            state_dict2 = torch.load(nets_path, map_location=map_location)
            self.netS.load_state_dict(state_dict2)

        print("The architecture of the substitute model:", s_net_name)

    def set_netG(self):
        if self.args.dataset == 'MNIST':
            self.netG = NetGenMnist(z_dim=self.z_dim).to(self.device)
        elif self.args.dataset == 'FashionMNIST':
            self.netG = NetGenMnist(z_dim=self.z_dim).to(self.device)
            # self.netG = NetGen(nz=self.z_dim, nc=1, img_size=28).to(self.device)
        else:  # CIFAR10, SVHN
            self.netG = NetGen(nz=self.z_dim, nc=3, img_size=32).to(self.device)

    @staticmethod
    def cross_entropy(q, p):
        return torch.mean(-torch.sum(p * torch.log(q+1e-8), dim=1))

    def shuffle_samples(self, data):
        # index = torch.randperm(data.shape[0])
        # res = data[index]
        res = data
        return [res[c:c + self.args.batch_size_g] for c in range(0, res.shape[0], self.args.batch_size_g)]

    def get_init_balanced_noise(self, gen_net, num_class=10, factor=1):
        num_total = self.args.batch_size_g * self.args.batch_num_g
        num_each = int(num_total / num_class)
        count_each = torch.zeros(10).long()
        if self.args.cuda:
            count_each = count_each.cuda()

        res_noise = None

        r = 0
        while True:
            if self.args.cuda:
                noise = torch.randn(1000, self.z_dim).cuda()
            else:
                noise = torch.randn(1000, self.z_dim).cpu()
            x_query = gen_net(noise)

            with torch.no_grad():
                v_output = self.msd.netV(x_query)
                v_output_p = F.softmax(v_output, dim=1)
                v_confidence, v_predicted = torch.max(v_output_p, 1)

            idx_c, cc = torch.unique(v_predicted, return_counts=True)

            for i, e in enumerate(count_each, 0):
                if e < num_each:
                    idx_i = (v_predicted == i).nonzero()
                    idx_i = idx_i.view(idx_i.shape[0])
                    e = e.int()

                    if res_noise is None:
                        res_noise = noise[idx_i][:num_each-e]
                    else:
                        res_noise = torch.cat((res_noise, noise[idx_i][:num_each-e]), 0)
            count_each[idx_c] += cc
            if r % 20 == 0:
                print("generated samples:", count_each)
                print("progress:", res_noise.shape[0], '/', num_total)
            r += 1
            if res_noise.shape[0] >= num_total * factor:
                break
        num_remainder = num_total - res_noise.shape[0]
        if num_remainder > 0:
            if self.args.cuda:
                noise = torch.randn(num_remainder, self.z_dim).cuda()
            else:
                noise = torch.randn(num_remainder, self.z_dim).cpu()
            res_noise = torch.cat((res_noise, noise), 0)
        print("generation rounds:", r, ", init query number:", r*1000+num_remainder)
        comm.save_results("ini_num_query = " + str(r*1000+num_remainder), self.args.res_filename)

        index = torch.randperm(res_noise.shape[0])
        res_noise = res_noise[index]

        res = gen_net(res_noise)
        return res, res_noise

    def train_netS(self, path_s, path_g=None, data_type="REAL", update_g=False, label_only=False, factor=1):
        if data_type == "REAL":
            self.train_netS_real(path_s, label_only)
        elif data_type == "NOISE":
            if not update_g:
                self.train_netS_noise_static_g(path_s, path_g, label_only)
            else:
                self.train_netS_noise_dynamic_g(path_s, path_g, label_only, factor=factor)
        else:
            print("Wrong training data type. It should be \'REAL\' or \'NOISE\'")

    def train_netS_noise_static_g(self, path_s, path_g, label_only=False):
        """
        Training the substitute net using synthetic noise samples to query.
        In this method, net G is no updated during the training.
        """
        print("Starting training net S using the setting \'noise_static_g\'")
        if self.args.cuda:
            self.netG = NetGenMnist(z_dim=self.z_dim).cuda()
        else:
            self.netG = NetGenMnist(z_dim=self.z_dim).cpu()
        self.netG.apply(self.weights_init)

        # optimizer_s = torch.optim.Adam(self.netS.parameters(), lr=self.args.lr)
        optimizer_s = torch.optim.RMSprop(self.netS.parameters(), lr=self.args.lr)

        self.netG.train()
        self.netS.train()
        self.msd.netV.eval()

        criterion = nn.CrossEntropyLoss()

        query_samples, _ = self.get_init_balanced_noise(self.netG)

        cc = torch.zeros(10).long()

        for epoch in range(self.args.epoch_g):
            print("epoch: %d/%d" % (epoch + 1, self.args.epoch_g))

            query_batches = self.shuffle_samples(query_samples)

            for i, x_query in enumerate(query_batches, 0):

                # Updating netS
                self.netS.zero_grad()

                with torch.no_grad():
                    v_output = self.msd.netV(x_query)
                    v_output_p = F.softmax(v_output, dim=1)
                    v_confidence, v_predicted = torch.max(v_output_p, 1)

                s_output = self.netS(x_query.detach())
                s_prob = F.softmax(s_output, dim=1)

                if label_only:
                    loss_s = criterion(s_output, v_predicted)
                    # loss_s = self.cross_entropy(s_prob, F.softmax(F.one_hot(v_predicted, 10).float(), dim=1))
                else:
                    loss_s = self.cross_entropy(s_prob, v_output_p)

                loss_s.backward()
                optimizer_s.step()

                if epoch == 0:
                    idx_c, cl = torch.unique(v_predicted, return_counts=True)
                    cc[idx_c] += cl

                if i % 200 == 0:
                    print("epoch", epoch+1, "batch idx:", i, "loss_s:", loss_s.detach().numpy())
                    print(v_predicted.cpu().detach().numpy().tolist())
                    print(v_confidence.cpu().detach().numpy().tolist())
                    print("cc:", cc)
                    # comm.accuracy(self.netS, 'netS', test_loader=self.test_loader)

        torch.save(self.netS.state_dict(), path_s)
        torch.save(self.netG.state_dict(), path_g)
        print("Finished training of netS")

    def initialize_netS(self, query_samples, optimizer_s, label_only):
        self.netG.train()
        self.netS.train()
        self.msd.netV.eval()

        criterion = nn.CrossEntropyLoss()

        train_loss_list = []
        acc_list = []

        for epoch in range(self.args.epoch_dg_ini):
            print("init netS epoch: %d/%d" % (epoch + 1, self.args.epoch_dg_ini))

            train_loss = 0
            train_correct = 0
            train_total = 0

            query_batches = self.shuffle_samples(query_samples)

            for i, x_query in enumerate(query_batches, 0):
                self.netS.zero_grad()

                with torch.no_grad():
                    v_output = self.msd.netV(x_query)
                    v_output_p = F.softmax(v_output, dim=1)
                    v_confidence, v_predicted = torch.max(v_output_p, 1)

                s_output = self.netS(x_query.detach())
                s_prob = F.softmax(s_output, dim=1)
                s_confidence, s_predicted = torch.max(s_prob, 1)

                if label_only:
                    loss_s = criterion(s_output, v_predicted)
                else:
                    loss_s = self.cross_entropy(s_prob, v_output_p)

                train_loss += loss_s.item()

                loss_s.backward()
                optimizer_s.step()
                if i == len(query_batches) - 1:
                    print("batch idx:", i, "loss_s:", loss_s.cpu().detach().numpy())
                    # print(v_predicted.cpu().detach().numpy().tolist())
                    # print(v_confidence.cpu().detach().numpy().tolist())
                    # print(torch.max(s_prob, 1)[0].cpu().detach().numpy())

            acc = comm.accuracy(self.netS, 'netS', test_loader=self.test_loader, cuda=self.args.cuda)
            acc_list.append(acc)
            train_loss_list.append(train_loss/len(query_batches))

        comm.save_results("ini_acc_list = " + str(acc_list), self.args.res_filename)
        comm.save_results("ini_train_loss_list = " + str(train_loss_list), self.args.res_filename)

    def train_netS_noise_dynamic_g(self, path_s, path_g, label_only=False, factor=1):
        """
        Training the substitute net using synthetic noise samples to query.
        In this method, net G is no updated during the training.
        """
        print("Starting training net S using the setting \'noise_dynamic_g\'")
        self.set_netG()

        if self.args.dataset == 'MNIST' or self.args.dataset == 'FashionMNIST':

            optimizer_s_ini = torch.optim.RMSprop(self.netS.parameters(), lr=self.args.lr)
            optimizer_s = torch.optim.RMSprop(self.netS.parameters(), lr=self.args.lr_tune_s)
            optimizer_g = torch.optim.RMSprop(self.netG.parameters(), lr=self.args.lr_tune_g)

            scheduler = torch.optim.lr_scheduler.StepLR(optimizer_s, step_size=self.args.decay_step, gamma=self.args.decay_gamma)

            qs, query_noise = self.get_init_balanced_noise(self.netG, factor=factor)
            self.initialize_netS(qs, optimizer_s_ini, label_only)
            query_batches = self.shuffle_samples(qs)

        else:  # CIFAR10

            optimizer_s_ini = torch.optim.RMSprop(self.netS.parameters(), lr=self.args.lr)
            optimizer_s = torch.optim.RMSprop(self.netS.parameters(), lr=self.args.lr_tune_s)
            optimizer_g = torch.optim.RMSprop(self.netG.parameters(), lr=self.args.lr_tune_g)

            scheduler = torch.optim.lr_scheduler.StepLR(optimizer_s, step_size=self.args.decay_step, gamma=self.args.decay_gamma)

            qs, query_noise = self.get_init_balanced_noise(self.netG, factor=factor)
            self.initialize_netS(qs, optimizer_s_ini, label_only)
            query_batches = self.shuffle_samples(qs)

        criterion = nn.CrossEntropyLoss()

        train_loss_list = []
        acc_list = []
        num_query_list = []

        torch.save(query_noise, "temp/query_noise.pt")
        torch.save(self.netS.state_dict(), "temp/netS_start.pth")
        torch.save(self.netG.state_dict(), "temp/netG_start.pth")

        for epoch in range(self.args.epoch_dg):
            print("***********************")
            print("global epoch: %d/%d" % (epoch + 1, self.args.epoch_dg))
            print('global epoch {0} lr: {1}'.format(epoch, optimizer_s.param_groups[0]['lr']))

            train_loss = 0
            train_correct = 0
            train_total = 0
            count_num_b = 0

            cc = torch.zeros(10).long()
            if self.args.cuda:
                cc = cc.cuda()

            # with torch.no_grad():
            #     query_samples = self.netG(query_noise)

            for epoch_s in range(self.args.epoch_dg_s):
                print("epoch_s: %d/%d" % (epoch_s + 1, self.args.epoch_dg_s))

                # query_batches = self.shuffle_samples(query_samples)
                for i, x_query in enumerate(query_batches, 0):
                    # Updating netS
                    self.netS.zero_grad()

                    with torch.no_grad():
                        v_output = self.msd.netV(x_query)
                        v_output_p = F.softmax(v_output, dim=1)
                        v_confidence, v_predicted = torch.max(v_output_p, 1)

                    s_output = self.netS(x_query.detach())
                    s_prob = F.softmax(s_output, dim=1)
                    s_confidence, s_predicted = torch.max(s_prob, 1)

                    if label_only:
                        loss_s = criterion(s_output, v_predicted)
                    else:
                        loss_s = self.cross_entropy(s_prob, v_output_p)

                    count_num_b += 1
                    train_loss += loss_s.item()
                    train_total += x_query.size(0)
                    train_correct += s_predicted.eq(v_predicted).sum().item()

                    loss_s.backward()
                    # nn.utils.clip_grad_norm_(self.netS.parameters(), max_norm=2.0, norm_type=2)
                    optimizer_s.step()

                    if epoch_s == 0:
                        idx_c, cl = torch.unique(v_predicted, return_counts=True)
                        cc[idx_c] += cl

                    if i == len(query_batches)-1:
                        print("batch idx:", i, "loss_s:", loss_s.cpu().detach().numpy())
                        print("cc:", cc)
                        comm.accuracy(self.netS, 'netS', test_loader=self.test_loader, cuda=self.args.cuda)
                        print("train_loss", (train_loss / count_num_b))
                        print("***")

            # save results in lists
            acc = comm.accuracy(self.netS, 'netS', test_loader=self.test_loader, cuda=self.args.cuda)
            acc_list.append(acc)
            train_loss_list.append(train_loss / count_num_b)
            num_query_list.append(query_noise.shape[0] * (epoch+1))

            # decayed learning rate
            scheduler.step()

            for epoch_g in range(self.args.epoch_dg_g):
                print("epoch_g: %d/%d" % (epoch_g + 1, self.args.epoch_dg_g))

                noise_batches = self.shuffle_samples(query_noise)
                for i, noise in enumerate(noise_batches, 0):
                    self.netG.zero_grad()

                    x_query = self.netG(noise)
                    s_output = self.netS(x_query)
                    s_output_p = F.softmax(s_output, dim=1)

                    # loss_g = - 0.1 * self.cross_entropy(s_output_p, s_output_p)
                    con, _ = torch.max(s_output_p, dim=1)
                    loss_g = - self.args.hyper_g * torch.mean(torch.log(con))

                    loss_g.backward()
                    optimizer_g.step()

                    if i % 100 == 0:
                        print("batch idx:", i, "loss_g:", loss_g.cpu().detach().numpy())

        # save results into the file
        comm.save_results("acc_list = " + str(acc_list), self.args.res_filename)
        comm.save_results("train_loss_list = " + str(train_loss_list), self.args.res_filename)
        comm.save_results("num_query_list = " + str(num_query_list), self.args.res_filename)

        torch.save(self.netS.state_dict(), path_s)
        torch.save(self.netG.state_dict(), path_g)
        print("Finished training of netS")

    def real_data_confidence(self):
        if self.args.dataset == "CIFAR10":
            transform_test = torchvision.transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform_test)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=1024, shuffle=False, num_workers=2)
            batch = next(iter(test_loader))
            input, target = batch
            self.msd.netV.eval()

            with torch.no_grad():
                v_output = self.msd.netV(input)
                v_output_p = F.softmax(v_output, dim=1)
                _, v_predicted = torch.max(v_output_p, 1)

            # print(v_predicted.cpu().detach().numpy().tolist())
            # print(_.cpu().detach().numpy().tolist())
            res = _.cpu().detach().numpy().tolist()
            r = sum(res)/len(res)
            print("averaged confidence on real data is:", r)


    def train_netS_real(self, path_s, label_only):
        """
        Training the substitute net using real samples to query.
        """
        print("Starting training net S using real samples to query.")

        # optimizer_s = torch.optim.Adam(self.netS.parameters(), lr=self.args.lr)
        optimizer_s = torch.optim.RMSprop(self.netS.parameters(), lr=self.args.lr)
        criterion = nn.CrossEntropyLoss()

        self.netS.train()
        self.msd.netV.eval()

        for epoch in range(self.args.epoch_g):
            print("epoch: %d/%d" % (epoch + 1, self.args.epoch_g))

            for i, data in enumerate(self.train_loader, 0):
                # Updating netS
                self.netS.zero_grad()

                x_query, _ = data

                with torch.no_grad():
                    v_output = self.msd.netV(x_query)
                    v_output_p = F.softmax(v_output, dim=1)
                    _, v_predicted = torch.max(v_output_p, 1)

                s_output = self.netS(x_query.detach())
                s_prob = F.softmax(s_output, dim=1)

                if label_only:
                    loss_s = criterion(s_output, v_predicted)
                else:
                    loss_s = self.cross_entropy(s_prob, v_output_p)

                loss_s.backward()
                optimizer_s.step()

                if i % 200 == 0:
                    print("batch idx:", i, "loss_s:", loss_s.cpu().detach().numpy())

                    print(v_predicted.cpu().detach().numpy().tolist())
                    print(_.cpu().detach().numpy().tolist())

        torch.save(self.netS.state_dict(), path_s)
        print("Finished training of netS")

    def create_adversary(self, method, targeted):
        self.adversary = self.get_adversary(method, targeted)
        self.netS.eval()

    def perturb(self, inputs, labels):
        if self.adversary is None:
            return inputs
        return self.adversary.perturb(inputs, labels)

    def attack(self, method="Clean", targeted=False):
        self.create_adversary(method, targeted)

        correct = 0.0
        total = 0.0

        for data in self.test_loader:
            inputs, labels = data

            if self.args.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            else:
                inputs = inputs.cpu()
                labels = labels.cpu()

            if targeted:
                if self.args.cuda:
                    # target_labels = torch.randint(0, 9, (inputs.shape[0],)).cuda()
                    target_labels = torch.ones((inputs.shape[0],), dtype=torch.int64).cuda()
                else:
                    # target_labels = torch.randint(0, 9, (inputs.shape[0],)).cpu()
                    target_labels = torch.ones((inputs.shape[0],), dtype=torch.int64).cpu()
                adv_inputs = self.perturb(inputs, target_labels)
                with torch.no_grad():
                    normal_outputs = self.msd.netV(inputs)
                    _, normal_predicted = torch.max(normal_outputs.data, 1)

                    outputs = self.msd.netV(adv_inputs)
                    _, predicted = torch.max(outputs.data, 1)

                    non_iden = (normal_predicted != target_labels)
                    adv_iden = (predicted == target_labels)
                    correct += (non_iden & adv_iden).sum() # succ attacking
                    total += non_iden.sum()
            else:
                adv_inputs = self.perturb(inputs, labels)
                with torch.no_grad():
                    normal_outputs = self.msd.netV(inputs)
                    _, normal_predicted = torch.max(normal_outputs.data, 1)

                    outputs = self.msd.netV(adv_inputs)
                    _, predicted = torch.max(outputs.data, 1)

                    correct_normal = (normal_predicted == labels)
                    identical_predict = (predicted == normal_predicted)
                    correct_adv_cur_batch = (correct_normal & identical_predict)

                    total += float(correct_normal.sum())
                    correct += correct_adv_cur_batch.sum()

        if targeted:
            print('Attack success rate (targeted) of \'%s\': %.2f %%' % (method, (100. * float(correct) / total)))
        else:
            print('Attack success rate (untargeted) of \'%s\': %.2f %%' % (method, (100 - 100. * float(correct) / total)))

    def weights_init(self, m):
        class_name = m.__class__.__name__
        if class_name.find('Conv') != -1:
            if self.args.cuda:
                m.weight.data = torch.where(m.weight.data > 0, m.weight.data, torch.zeros(m.weight.data.shape).cuda())
            else:
                m.weight.data = torch.where(m.weight.data > 0, m.weight.data, torch.zeros(m.weight.data.shape))



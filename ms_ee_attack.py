from ms_attack import *
from ms_grad_approx import estimate_gradient

from ms_nets import NetGenSMnist, NetGenEMnist, NetGenS, NetGenE


class MSEEAttack(MSAttack):

    def __init__(self, args, defense_obj=None):
        super(MSEEAttack, self).__init__(args, defense_obj)
        self.netGE = None

    def set_netG(self):
        if self.args.dataset == 'MNIST':
            self.netG = NetGenSMnist(z_dim=self.z_dim).to(self.device)
            self.netGE = NetGenEMnist().to(self.device)
        elif self.args.dataset == 'FashionMNIST':
            self.netG = NetGenSMnist(z_dim=self.z_dim).to(self.device)
            self.netGE = NetGenEMnist().to(self.device)
        else:  # CIFAR10, SVHN
            self.netG = NetGenS(nz=self.z_dim, nc=3, img_size=32).to(self.device)
            self.netGE = NetGenE(nz=self.z_dim, nc=3, img_size=32, act=torch.tanh).to(self.device)

    def ee_train_netS(self, path_s, path_g=None, path_ge=None):
        """
        Training the substitute net using Exploration and Exploitation.
        """
        print("Starting training net S using \'Exploration and Exploitation\'")

        self.set_netG()
        if self.args.dataset == 'MNIST' or self.args.dataset == 'FashionMNIST' or self.args.dataset == 'SVHN':
            self.netG.apply(self.weights_init)
            self.netGE.apply(self.weights_init)

        optimizer_s = torch.optim.RMSprop(self.netS.parameters(), lr=self.args.lr_tune_s)
        optimizer_g = torch.optim.RMSprop(self.netG.parameters(), lr=self.args.lr_tune_g)
        optimizer_ge = torch.optim.RMSprop(self.netGE.parameters(), lr=self.args.lr_tune_ge)
        # optimizer_ge = torch.optim.RMSprop(self.netGE.parameters(), lr=0.0)

        steps = sorted([int(step * self.args.epoch_dg) for step in self.args.steps])
        print("Learning rate scheduling at steps: ", steps)
        scheduler_s = torch.optim.lr_scheduler.MultiStepLR(optimizer_s, steps, self.args.scale)
        scheduler_g = torch.optim.lr_scheduler.MultiStepLR(optimizer_g, steps, self.args.scale)
        scheduler_ge = torch.optim.lr_scheduler.MultiStepLR(optimizer_ge, steps, self.args.scale)

        ce_criterion = nn.CrossEntropyLoss()

        acc_list = []
        num_query_list = []
        query_num = 0

        torch.save(self.netS.state_dict(), path_s + "_start.pth")
        torch.save(self.netG.state_dict(), path_g + "_start.pth")
        torch.save(self.netGE.state_dict(), path_ge + "_start.pth")

        best_acc = 0
        for epoch in range(self.args.epoch_dg):
            print("***********************")
            print("global epoch: %d/%d" % (epoch + 1, self.args.epoch_dg))
            print('global epoch {0} lr_s: {1} lr_g: {2} lr_ge: {3}'.format(epoch + 1, optimizer_s.param_groups[0]['lr'],
                                                optimizer_g.param_groups[0]['lr'], optimizer_ge.param_groups[0]['lr']))

            for e_iter in range(self.args.epoch_itrs):

                for ng in range(self.args.epoch_dg_g):
                    self.netG.zero_grad()

                    with torch.no_grad():
                        noise = torch.randn(self.args.batch_size_g, self.z_dim).to(self.device)

                    if self.args.l_only:
                        x_query = self.netGE(self.netG(noise))
                        s_output = self.netS(x_query)

                        with torch.no_grad():
                            v_output = self.msd.netV(x_query)
                            v_output_p = F.softmax(v_output, dim=1)
                            v_confidence, v_predicted = torch.max(v_output_p, 1)

                        query_num += x_query.shape[0]

                        loss_g = - ce_criterion(s_output, v_predicted)

                        loss_g.backward()
                        optimizer_g.step()
                    else:
                        z_temp = self.netG(noise)
                        x_query = self.netGE(z_temp, no_act=True)
                        approx_grad_wrt_x, loss_g, q_num = estimate_gradient(self.args, self.msd.netV, self.netS, x_query,
                                                                             loss_type="l1", act="tanh")

                        query_num += q_num

                        x_query.backward(approx_grad_wrt_x)

                        optimizer_g.step()

                    if e_iter == self.args.epoch_itrs - 1:
                        print("loss_g:", loss_g.cpu().detach().numpy())

                for ii in range(self.args.epoch_exploit):
                    for ne in range(self.args.epoch_dg_ge):
                        self.netGE.zero_grad()

                        with torch.no_grad():
                            noise = torch.randn(self.args.batch_size_g, self.z_dim).to(self.device)

                        x_seed = self.netG(noise).detach()
                        x_query = self.netGE(x_seed)

                        if self.args.l_only:
                            s_output = self.netS(x_query)
                            s_output_p = F.softmax(s_output, dim=1)

                            con, _ = torch.max(s_output_p, dim=1)
                            loss_ge = - torch.mean(torch.log(con))

                            loss_ge.backward()
                            optimizer_ge.step()
                        else:
                            approx_grad_wrt_x, loss_ge, q_num = estimate_gradient(self.args, self.msd.netV, self.netS, x_query,
                                                                                  loss_type="confidence", act="tanh")
                            query_num += q_num

                            x_query.backward(approx_grad_wrt_x)

                            optimizer_ge.step()
                        if e_iter == self.args.epoch_itrs - 1:
                            print("loss_ge:", loss_ge.cpu().detach().numpy())
                    for ns in range(self.args.epoch_dg_s):
                        self.netS.zero_grad()

                        with torch.no_grad():
                            noise = torch.randn(self.args.batch_size_g, self.z_dim).to(self.device)

                        x_query = self.netGE(self.netG(noise)).detach()
                        s_output = self.netS(x_query)
                        # s_output_p = F.softmax(s_output, dim=1)

                        with torch.no_grad():
                            v_output = self.msd.netV(x_query)
                            v_output_p = F.softmax(v_output, dim=1)

                        query_num += x_query.shape[0]

                        if self.args.l_only:
                            v_confidence, v_predicted = torch.max(v_output_p, 1)
                            loss_s = ce_criterion(s_output, v_predicted)
                        else:
                            v_logit = v_output
                            v_logit = F.log_softmax(v_logit, dim=1).detach()
                            v_logit -= v_logit.mean(dim=1).view(-1, 1).detach()
                            loss_s = F.l1_loss(s_output, v_logit)

                        loss_s.backward()
                        optimizer_s.step()

                        if (e_iter == self.args.epoch_itrs - 1) and (ns == self.args.epoch_dg_s - 1):
                            print("ns idx:", ns, "loss_s:", loss_s.cpu().detach().numpy())

            scheduler_s.step()
            scheduler_g.step()
            scheduler_ge.step()

            # save results in lists
            acc = comm.accuracy(self.netS, 'netS', test_loader=self.test_loader, cuda=self.args.cuda)
            acc_list.append(acc)
            num_query_list.append(query_num)

            if acc > best_acc:
                torch.save(self.netS.state_dict(), path_s + ('_epoch_%d.pth' % epoch))
                torch.save(self.netG.state_dict(), path_g + ('_epoch_%d.pth' % epoch))
                torch.save(self.netGE.state_dict(), path_ge + ('_epoch_%d.pth' % epoch))
                best_acc = acc

        # save results into the file
        scenario = "_p_only"
        if self.args.l_only:
            scenario = "_l_only"
        comm.save_results("acc_list = " + str(acc_list), self.args.res_filename + scenario)
        comm.save_results("num_query_list = " + str(num_query_list), self.args.res_filename + scenario)

        # save the final model
        torch.save(self.netS.state_dict(), path_s + "_over.pth")
        torch.save(self.netG.state_dict(), path_g + "_over.pth")
        torch.save(self.netGE.state_dict(), path_ge + "_over.pth")
        print("Finished training of netS")



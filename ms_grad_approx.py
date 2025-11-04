import numpy as np
import torch
import torch.nn.functional as F


def estimate_gradient(args, victim_model, clone_model, x, epsilon=1e-3, m=1, num_classes=10, loss_type="l1", act="relu"):

    if args.cuda:
        device = "cuda:0"
    else:
        device = "cpu"

    if act == "relu":
        g_activation = torch.relu
    else:
        g_activation = torch.tanh

    clone_model.eval()
    victim_model.eval()
    with torch.no_grad():
        # Sample unit noise vector
        N = x.size(0)
        C = x.size(1)
        S = x.size(2)
        dim = S ** 2 * C

        u = np.random.randn(N * m * dim).reshape(-1, m, dim)  # generate random points from normal distribution

        d = np.sqrt(np.sum(u ** 2, axis=2)).reshape(-1, m, 1)  # map to a uniform distribution on a unit sphere
        u = torch.Tensor(u / d).view(-1, m, C, S, S)
        u = torch.cat((u, torch.zeros(N, 1, C, S, S)), dim=1)  # Shape N, m + 1, S^2

        u = u.view(-1, m + 1, C, S, S)

        evaluation_points = (x.view(-1, 1, C, S, S).cpu() + epsilon * u).view(-1, C, S, S)
        evaluation_points = g_activation(evaluation_points)

        # Compute the approximation sequentially to allow large values of m
        pred_victim = []
        pred_clone = []
        max_number_points = 32 * 156  # Hardcoded value to split the large evaluation_points tensor to fit in GPU

        for i in (range(N * m // max_number_points + 1)):
            pts = evaluation_points[i * max_number_points: (i + 1) * max_number_points]
            pts = pts.to(device)

            pred_victim_pts = victim_model(pts).detach()
            pred_clone_pts = clone_model(pts)

            pred_victim.append(pred_victim_pts)
            pred_clone.append(pred_clone_pts)

        pred_victim = torch.cat(pred_victim, dim=0).to(device)
        pred_clone = torch.cat(pred_clone, dim=0).to(device)

        query_num = pred_victim.shape[0]

        u = u.to(device)

        if loss_type == "l1":
            pred_victim = F.log_softmax(pred_victim, dim=1).detach()
            pred_victim -= pred_victim.mean(dim=1).view(-1, 1).detach()

            loss_fn = F.l1_loss
            loss_values = - loss_fn(pred_clone, pred_victim, reduction='none').mean(dim=1).view(-1, m + 1)
        elif loss_type == "cross_entropy":
            pred_clone = F.softmax(pred_clone, dim=1)
            pred_victim = F.softmax(pred_victim.detach(), dim=1)

            loss_fn = cross_entropy
            loss_values = - loss_fn(pred_clone, pred_victim).sum(dim=1).view(-1, m + 1)
        elif loss_type == "kl_loss":
            pred_clone = F.softmax(pred_clone, dim=1)
            pred_victim = F.softmax(pred_victim.detach(), dim=1)

            loss_fn = kl_loss
            loss_values = - loss_fn(pred_clone, pred_victim).sum(dim=1).view(-1, m + 1)
        elif loss_type == "confidence":
            pred_victim = F.softmax(pred_victim.detach(), dim=1)
            conf, _ = torch.max(pred_victim, dim=1)
            loss_values = conf.view(-1, m + 1)
        else:
            loss_values = None

        # Compute difference following each direction
        differences = loss_values[:, :-1] - loss_values[:, -1].view(-1, 1)
        differences = differences.view(-1, m, 1, 1, 1)

        # Formula for Forward Finite Differences
        gradient_estimates = 1 / epsilon * differences * u[:, :-1]
        gradient_estimates *= dim

        if loss_type == "l1" or loss_type == "confidence":
            gradient_estimates = gradient_estimates.mean(dim=1).view(-1, C, S, S) / (num_classes * N)
        elif loss_type == "cross_entropy" or loss_type == "kl_loss":
            gradient_estimates = gradient_estimates.mean(dim=1).view(-1, C, S, S)
        else:
            gradient_estimates = None

        clone_model.train()
        loss_G = loss_values[:, -1].mean()
        return gradient_estimates.detach(), loss_G, query_num


def cross_entropy(q, p):
    return -p * torch.log(q+1e-8)

def kl_loss(q, p):
    return p * torch.log(p+1e-8) - p * torch.log(q+1e-8)

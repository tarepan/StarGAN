import time
import torch
import torch.nn.functional as F
import numpy as np


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out


def classification_loss(logit, target, dataset='CelebA'):
    """Compute binary or softmax cross entropy loss."""
    if dataset == 'CelebA':
        return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
    elif dataset == 'RaFD':
        return F.cross_entropy(logit, target)


def train(config, G, D, g_optimizer, d_optimizer, data_loader, device, writer):
    """Train StarGAN within a single dataset."""
    # Learning rate cache for decaying.
    g_lr = config.g_lr
    d_lr = config.d_lr

    # Start training.
    print('Start training...')
    start_time = time.time()
    start_iters = 0
    data_iter = iter(data_loader)
    for i in range(start_iters, config.num_iters):
        # single iteration
        ####### 1. Preprocess input data
        # Fetch real images and labels.
        x_real, label_org = next(data_iter)

        # Generate target domain labels randomly.
        rand_idx = torch.randperm(label_org.size(0))
        label_trg = label_org[rand_idx]

        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if config.dataset == 'CelebA':
            c_org = label_org.clone()
            c_trg = label_trg.clone()
        elif config.dataset == 'RaFD':
            c_org = label2onehot(label_org, config.c_dim)
            c_trg = label2onehot(label_trg, config.c_dim)
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        x_real = x_real.to(device)           # Input images.
        c_org = c_org.to(device)             # Original domain labels.
        c_trg = c_trg.to(device)             # Target domain labels.
        label_org = label_org.to(device)     # Labels for computing classification loss.
        label_trg = label_trg.to(device)     # Labels for computing classification loss.

        ####### 2. Train the discriminator

        # Compute loss with real images.
        out_src, out_cls = D(x_real)
        # mean of 2D probability patch
        d_loss_real = - torch.mean(out_src)
        d_loss_cls = classification_loss(out_cls, label_org, config.dataset)

        # Compute loss with fake images.
        x_fake = G(x_real, c_trg)
        out_src, out_cls = D(x_fake.detach())
        d_loss_fake = torch.mean(out_src)

        # Compute loss for gradient penalty.
        alpha = torch.rand(x_real.size(0), 1, 1, 1).to(device)
        x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
        out_src, _ = D(x_hat)
        d_loss_gp = gradient_penalty(out_src, x_hat, device)

        # Backward and optimize.
        d_loss = d_loss_real + d_loss_fake + config.lambda_cls * d_loss_cls + config.lambda_gp * d_loss_gp
        g_optimizer.zero_grad()
        d_optimizer.zero_grad()

        d_loss.backward()
        d_optimizer.step()

        loss = {}
        loss['D/adv_real'] = d_loss_real.item()
        loss['D/adv_fake'] = d_loss_fake.item()
        loss['D/cls'] = d_loss_cls.item()
        loss['D/gp'] = d_loss_gp.item()
        ########  3. Train the generator

        # 1 Generator training per n_critic Discriminator training
        if (i+1) % config.n_critic == 0:
            # Original-to-target domain.
            x_fake = G(x_real, c_trg)
            out_src, out_cls = D(x_fake)
            g_loss_fake = - torch.mean(out_src)
            g_loss_cls = classification_loss(out_cls, label_trg, config.dataset)

            # Target-to-original domain.
            x_reconst = G(x_fake, c_org)
            g_loss_rec = torch.mean(torch.abs(x_real - x_reconst)) # L1 loss

            # Backward and optimize.
            g_loss = g_loss_fake + config.lambda_rec * g_loss_rec + config.lambda_cls * g_loss_cls
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            loss['G/adv'] = g_loss_fake.item()
            loss['G/rec'] = g_loss_rec.item()
            loss['G/cls'] = g_loss_cls.item()
        # =================================================================================== #
        #                               4. After                                              #
        # =================================================================================== #

        # Decay learning rates.
        if (i+1) % config.lr_update_step == 0 and (i+1) > (config.num_iters - config.num_iters_decay):
            g_lr -= (config.g_lr / float(config.num_iters_decay))
            d_lr -= (config.d_lr / float(config.num_iters_decay))
            # """Decay learning rates of the generator and discriminator."""
            for param_group in g_optimizer.param_groups:
                param_group['lr'] = g_lr
            for param_group in d_optimizer.param_groups:
                param_group['lr'] = d_lr

        if (i+1) % config.log_step == 0:
            for tag, value in loss.items():
                writer.add_scalar(tag, value, i+1)

def gradient_penalty(y, x, device):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)

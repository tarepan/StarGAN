import os

import torch
from torchvision.utils import save_image

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def create_labels(c_org, c_dim=5, dataset='CelebA', selected_attrs=None, device):
    """Generate target domain labels for debugging and testing."""
    # Get hair color indices.
    if dataset == 'CelebA':
        hair_color_indices = []
        for i, attr_name in enumerate(selected_attrs):
            if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                hair_color_indices.append(i)

    c_trg_list = []
    for i in range(c_dim):
        if dataset == 'CelebA':
            c_trg = c_org.clone()
            if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                c_trg[:, i] = 1
                for j in hair_color_indices:
                    if j != i:
                        c_trg[:, j] = 0
            else:
                c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
        elif dataset == 'RaFD':
            c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)

        c_trg_list.append(c_trg.to(device))
    return c_trg_list

def test(config, G, data_loader, device):
    """Translate images using StarGAN trained on a single dataset."""
    # Load the trained generator.

    with torch.no_grad():
        for i, (x_real, c_org) in enumerate(data_loader):

            # Prepare input images and target domain labels.
            x_real = x_real.to(device)
            c_trg_list = self.create_labels(c_org, config.c_dim, config.dataset, config.selected_attrs, device)

            # Translate images.
            x_fake_list = [x_real]
            for c_trg in c_trg_list:
                x_fake_list.append(G(x_real, c_trg))

            # Save the translated images.
            x_concat = torch.cat(x_fake_list, dim=3)
            result_path = os.path.join(config.result_dir, '{}-images.jpg'.format(i+1))
            # """Convert the range from [-1, 1] to [0, 1]."""
            denorm = lambda x: ((x + 1) / 2).clamp_(0, 1)
            save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
            print('Saved real and fake images into {}...'.format(result_path))

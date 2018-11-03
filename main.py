import os
import argparse

import torch
from torch.backends import cudnn

from networks.Generator import Generator
from networks.Discriminator import Discriminator
from data_loader import getProperLoader
from trains.train import train


def main(config):
    # For fast training.
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # running preparation
    prepareDirs(config)

    # Data preparation
    data_loader = getProperLoader(config)

    # Model Initialization
    G = Generator(config.g_conv_dim, config.c_dim, config.g_repeat_num)
    D = Discriminator(config.image_size, config.d_conv_dim, config.c_dim, config.d_repeat_num)
    g_optimizer = torch.optim.Adam(G.parameters(), config.g_lr, [config.beta1, config.beta2])
    d_optimizer = torch.optim.Adam(D.parameters(), config.d_lr, [config.beta1, config.beta2])
    G.to(device)
    D.to(device)

    # Training/Test
    if config.mode == 'train':
        train(config, G, D, g_optimizer, d_optimizer, data_loader, device)
    elif config.mode == 'test':
        test(config, G, data_loader, device)


def importAndProcessConfigs(parser, name="NNconfig.json"):
    import json
    f = open(name, "r")
    argSettings = json.load(f)
    f.close()
    for set in argSettings:
        # extract positional argments
        posArgs = []
        posArgKeys = ["name", "list"]
        posArgs.append("--" + set.pop("name"))
        if "list" in set:
            set.pop("list")
            posArgs.append("--list")
        # convert type
        if "type" in set:
            set["type"] = int if set["type"] == "int" else float if set["type"] == "float" else str if set["type"] == "str" else None
            assert set["type"] is not None
        if "default" in set and set["default"] == "None":
            set["default"] = None

        parser.add_argument(*posArgs, **set)


def loadConfig(name="NNconfig.json"):
    import json
    f = open(name, "r")
    argSettings = json.load(f)
    f.close()
    cfg = {}
    for set in argSettings:
        if "default" in set and set["default"] == "None":
            set["default"] = None
        cfg[set["name"]] = set["default"]
    return cfg

def prepareDirs(config):
    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    importAndProcessConfigs(parser)
    config = parser.parse_args()
    main(config)

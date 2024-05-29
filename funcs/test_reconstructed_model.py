import argparse
import os.path


import torch.multiprocessing as mp
import models.NWS as NWS

from dataset.cifar100_dataset import cifar100_val_loader
from dataset.fine_grained_dataset import ds_val_loader
from funcs.utils_funcs import *
from models.NwsMobileNet import *
from tqdm import tqdm
from models.NwsMLP import NwsMLP
from trainers.NwsTrainer import NwsTrainer
# cudnn.deterministic = True
from dataset.mnist_dataset import MNIST_val_loader
from funcs.setup import parse_args
from models.NWS import  load_fc_codes_from_lzma, load_fc_codes2weights_individual_, get_fc_info

@torch.no_grad()
def test_codes(loader, model, device):
    loader = tqdm(loader)
    acc_sum = 0
    n_sum = 0
    model.eval()
    # eval_mode(True) is necessary when reloading and test if model is constructed from codes;
    # eval_mode(False) is necessary when reloading and test if model load from ckpt and temporary weights;
    model.eval_mode(True)
    for x, y in loader:
        y = y.to(device)
        x = x.to(device)

        logits = model(x)
        logits = logits.view(logits.size(0), -1)

        _, winners = (logits).max(1)

        acc = torch.sum((winners == y).int())

        acc_sum += acc.detach().item()
        n_sum += y.size(0)
        avg_acc = acc_sum / n_sum

        loader.set_description(
            (
                f" acc:{avg_acc:.5f} ;"
            )
        )
    logging.info('====> Evaluation using reconstructed model: acc {}'.format(avg_acc))
    return avg_acc


def main(args):

    device = "cuda"

    arch = args.arch
    n_emb = args.n_emb
    num_outputs = args.end_class
    backbone = NwsMLP(28 * 28, 512, 512, n_emb=args.n_emb, n_classes=10)
    model = NwsTrainer(args, backbone)

    # reset num_emb to load the KPs of the previous model

    # way 1; load from temporary weights
    # ckpt = torch.load(args.ckpt_root)
    #
    # model = load_state_from_ddp(model, torch.load(args.ckpt_root)['state_dict'])

    # way 2; load from codes
    qtzs_in_order = torch.load(f"{args.ckpt_root}/meta_weights.pt")

    bns = torch.load(f"{args.ckpt_root}/task{args.task_id}_bns.pt")
    codes_path = f'{args.ckpt_root}/{args.task_id}task_codes.lzma'
    fc_info = get_fc_info(model)
    codes_layers = load_fc_codes_from_lzma(codes_path, fc_info)
    load_fc_codes2weights_individual_(model, codes_layers, fc_info, qtzs_in_order, bns)

    model = model.cuda()

    if args.dataset == 'cifar':

        test_loader = cifar100_val_loader(args.dataset_path, args.task_id, args.batch_size, args.num_workers)

    elif args.dataset == 'mnist':

        test_loader = MNIST_val_loader(args.dataset_path, args.batch_size, args.num_workers)

    # eval_mode(True) is necessary when reloading and test if model is constructed from codes;
    # eval_mode(False) is necessary when reloading and test if model load from ckpt and temporary weights;
    # model.backbone.eval_mode(True)
    # test_codes(train_loader, model, device, args)
    test_codes(test_loader, backbone, device)

##########


if __name__ == "__main__":
    args = parse_args()
    # set random seed
    set_seed(args.seed)
    print(args)
    main(args)




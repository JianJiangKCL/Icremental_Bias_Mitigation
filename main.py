import warnings
warnings.filterwarnings("ignore")
from datasets.data_module import Modalities
from models.multi_modality_perceiver import MultiModalityPerceiver
from models.multimodal_infomax import MMIM
from funcs.build_dataset import get_loader
from funcs.setup import parse_args, set_logger, set_trainer
from funcs.utils_funcs import load_state_dict_flexible_, set_seed
import os
import wandb
from trainers.BaselineTrainer import BaselineTrainer
from trainers.MultiTaskTrainer import MultiTaskTrainer
from trainers.IncrementTrainer import IncrementTrainer
import torch
import copy


def main(args):
	name_modalities = args.modalities
	if len(name_modalities) == 1:
		# if contains , then split
		if '_' in name_modalities[0]:
			name_modalities = name_modalities[0].split('_')
	print(f"modalities: {name_modalities}")
	file_prefix = '_'.join(name_modalities)
	# if args.bias_sensitive is not None:
	# 	file_prefix = f"/bias_{args.bias_sensitive}_group{args.bias_group}_personality{args.bias_personality}/" + file_prefix
	file_suffix = f"_lr{args.lr}_e{args.epochs}_seed{args.seed}_opt{args.optimizer}_" \
						   f"bs{args.batch_size}_beta{args.beta}_alpha_{args.alpha}_gamma_{args.gamma}_beta_{args.beta}"

	if args.arch == 'infomax':
		file_suffix += f"_sigma_{args.sigma}_cpc{args.cpc_layers}_dropout_{args.dropout_prj}"

	if args.target_personality is not None:
		file_suffix += f"/personality_{args.target_personality}"

	root_dir = save_path = f"{args.results_dir}/{args.target_sensitive_group}/{file_prefix}{file_suffix}"
	os.makedirs(save_path, exist_ok=True)


	modalities = [Modalities[name] for name in name_modalities]
	if args.dataset == 'udiva':
		sensitive_groups = ["gender", "age"]
	elif args.dataset == 'fiv2':
		sensitive_groups = ["gender", "ethnicity"]
	train_loader = get_loader(args, name_modalities, sensitive_groups,  'train_val') #'train')#
	val_loader = get_loader(args, name_modalities, sensitive_groups, 'test') #'validation_test')#

	test_loader = get_loader(args, name_modalities, sensitive_groups,  'test') #'validation_test')#

	if args.arch == 'perceiver':
		backbone = MultiModalityPerceiver(
			modalities=modalities,
			depth=args.depth,
			num_latents=args.num_latents,
			latent_dim=args.latent_dim,
			cross_heads=args.cross_heads,
			latent_heads=args.latent_heads,
			cross_dim_head=args.cross_dim_head,
			latent_dim_head=args.latent_dim_head,
			num_outputs=args.num_outputs,
			attn_dropout=0.,
			ff_dropout=0.,
			weight_tie_layers=True
		)

	elif args.arch == 'infomax':
		backbone = MMIM(args)

	if args.finetune:
		print('original finetune: ', args.finetune)
		if 'ttt' in args.finetune:
			# fintune from the previously debiased sensitive group
			# sensitive_groups = ["gender", "age"]
			# remove target sensitive group from sensitive_groups and assign to tmp
			if args.test_only:
				tmp = [args.target_sensitive_group]
			else:
				tmp = sensitive_groups.copy()
				tmp.remove(args.target_sensitive_group)
			args.finetune = args.finetune.replace('ttt', tmp[0])
		if 'xxx' in args.finetune:
			args.finetune = args.finetune.replace('xxx', file_prefix)
		if 'sss' in args.finetune:
			args.finetune = args.finetune.replace('sss', str(args.seed))
		if 'ppp' in args.finetune:
			args.finetune = args.finetune.replace('ppp', str(args.target_personality))
		print(f"finetune from {args.finetune}")
		checkpoint = torch.load(args.finetune)
		backbone = load_state_dict_flexible_(backbone, checkpoint['state_dict'])

	if args.is_incremental:
		old_model = copy.deepcopy(backbone)
		model = IncrementTrainer(args, backbone, old_model, name_modalities, sensitive_groups)
	else:
		Trainer = BaselineTrainer if args.is_baseline else MultiTaskTrainer
		model = Trainer(args, backbone, name_modalities, sensitive_groups)

	logger = None
	if args.use_logger:
		logger = set_logger(args, root_dir)
	trainer = set_trainer(args, logger, save_path)
	if not args.test_only:
		trainer.fit(model, train_loader, val_loader)
		print('--------------finish training')
		# trainer.save_checkpoint(f'{save_path}/checkpoint.pt')
	trainer.test(model, test_loader)
	wandb.finish()


if __name__ == "__main__":
	args = parse_args()
	# set random seed
	set_seed(args.seed)
	print(args)

	main(args)
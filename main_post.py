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
from trainers.baselines.PostProcTrainer import PostProcTrainer
from models.baselines.adversial_debias import adversary_model
import torch.nn as nn
from models.optimizers import Lamb
import torch.nn.functional as F
import torch

def main(args):
	name_modalities = args.modalities
	if len(name_modalities) == 1:
		# if contains , then split
		if '_' in name_modalities[0]:
			name_modalities = name_modalities[0].split('_')
	print(f"modalities: {name_modalities}")
	file_prefix = '_'.join(name_modalities)

	file_suffix = f"_lr{args.lr}_e{args.epochs}_seed{args.seed}_opt{args.optimizer}_" \
						   f"bs{args.batch_size}"


	if args.target_personality is not None:
		file_suffix += f"/personality_{args.target_personality}"

	root_dir = save_path = f"{args.results_dir}/{args.target_sensitive_group}/{file_prefix}{file_suffix}"
	os.makedirs(save_path, exist_ok=True)


	modalities = [Modalities[name] for name in name_modalities]
	if args.dataset == 'udiva':
		sensitive_groups = ["gender", "age"]
	elif args.dataset == 'fiv2':
		sensitive_groups = ["gender", "ethnicity"]
	# drop_last cannot solve the NAN caused by tp cost.
	train_loader = get_loader(args, name_modalities, sensitive_groups,  'train')
	val_loader = get_loader(args, name_modalities, sensitive_groups, 'validation')

	test_loader = get_loader(args, name_modalities, sensitive_groups,  'test')


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

	if args.finetune:
		print('original finetune: ', args.finetune)
		if 'ttt' in args.finetune:
			# fintune from the previously debiased sensitive group
			# sensitive_groups = ["gender", "age"]
			# remove target sensitive group from sensitive_groups and assign to tmp
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

	model = PostProcTrainer(args, backbone, name_modalities, sensitive_groups)

	logger = None
	if args.use_logger:
		logger = set_logger(args, root_dir)
	trainer = set_trainer(args, logger, save_path)

	#
	if not args.finetune:
		trainer.fit(model, train_loader, val_loader)
		print('--------------finish training')
		# trainer.save_checkpoint(f'{save_path}/checkpoint.pt')
	# calibrate the results for the first time
	model.set_postprocess(True)
	print('--------------start calibrating')
	trainer.validate(model, val_loader)
	trainer.test(model, test_loader)

	# calibrate the results for the second time
	if args.dataset != 'fiv2':
		model.set_incremental(True)
		next_target_sensitive_group = sensitive_groups.copy()
		next_target_sensitive_group.remove(args.target_sensitive_group)
		model.set_target_sensitive_group(next_target_sensitive_group[0])
		print('--------------start second calibrating')
		trainer.validate(model, val_loader)
		trainer.test(model, test_loader)


	wandb.finish()


if __name__ == "__main__":
	args = parse_args()
	# set random seed
	set_seed(args.seed)
	print(args)

	main(args)
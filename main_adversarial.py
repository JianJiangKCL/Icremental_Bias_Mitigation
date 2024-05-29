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
from trainers.baselines.AdversarialTrainer import AdversarialTrainer
import torch
from models.baselines.adversial_debias import adversary_model
import torch.nn as nn
from models.optimizers import Lamb
import torch.nn.functional as F


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
	train_loader = get_loader(args, name_modalities, sensitive_groups,  'train_val') #'train')#
	val_loader = get_loader(args, name_modalities, sensitive_groups, 'test') #'validation_test')#

	test_loader = get_loader(args, name_modalities, sensitive_groups,  'test') #'validation_test')#


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



	def init_parameters(net):
		for m in net.modules():
			if isinstance(m, nn.Linear):
				# nn.init.xavier_uniform(m.weight.data)
				torch.nn.init.normal_(m.weight.data)
				if m.bias is not None:
					nn.init.constant_(m.bias.data, 0)
	# pretrain classifier
	# backbone = backbone.cuda()
	# init_parameters(backbone)
	# mse_loss = nn.MSELoss()
	# clf_opt = Lamb(backbone.parameters(), lr=args.lr)
	# print('pretrain classifier')
	# backbone.train()
	# for epoch in range(args.epochs//2):
	# 	for batch in train_loader:
	# 		x, label_ocean, label_sen_dict = batch
	# 		modalities_x = {modality: x[modality].cuda() for modality in name_modalities}
	#
	# 		pred_ocean = backbone.forward(modalities_x)
	#
	# 		loss_mse = mse_loss(pred_ocean,
	# 		                         label_ocean[:, args.target_personality].cuda())
	# 		loss_mse.backward()
	# 		clf_opt.step()
	# 		clf_opt.zero_grad()
	#
	# print('pretrain classifier done')

	# pretrain adversial generator
	print('pretrain adversial generator')
	adv_model = adversary_model().cuda()
	init_parameters(adv_model)
	adv_opt = torch.optim.Adam(adv_model.parameters(), lr=0.001, weight_decay=1e-5)
	adv_loss = F.binary_cross_entropy_with_logits
	for epoch in range(10):
		# only pretrain the generator
		adv_model.train()
		backbone.eval()
		for batch in train_loader:
			x, label_ocean, label_sen_dict = batch

			y = label_sen_dict[args.target_sensitive_group].unsqueeze(1).cuda()
			y_s = label_sen_dict[args.target_sensitive_group].unsqueeze(1).float().cuda()

			# label_sen_dict = {key: value.cuda() for key, value in label_sen_dict.items()}
			modalities_x = {modality: x[modality].cuda() for modality in name_modalities}

			pred_ocean = backbone.forward(modalities_x)
			pred_ocean = pred_ocean[:, args.target_personality].unsqueeze(1)
			pred_protected_attributes_labels, pred_protected_attributes_logits = adv_model.forward(pred_ocean, y)
			# y_b = get_binary_ocean_values(label_ocean)
			loss = adv_loss(pred_protected_attributes_logits, y_s,
			                     reduction='mean')
			loss.backward()
			adv_opt.step()
			adv_opt.zero_grad()
	print('pretrain adversial generator done')

	# train
	backbone.train()
	adv_model.train()
	args.epochs = args.epochs - args.epochs//2
	adv_model = adversary_model()
	model = AdversarialTrainer(args, backbone, adv_model, name_modalities, sensitive_groups)

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
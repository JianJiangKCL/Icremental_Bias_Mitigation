from argparse import ArgumentParser
import argparse

# it cannot be triggered in normal yaml setting,
# because yaml replace the values of args, rather than assign values to args during parsing
# so this is only valid for sweep yaml setting which assign values to args
class ParserList(argparse.Action):
	def __call__(self, parser, namespace, values, option_string=None):
		# when nargs='+' is used, values is a list
		# if isinstance(values, str):
		# 	values = values.split('_')
		if len(values) == 1:
			# if contains , then split
			if '_' in values[0]:
				values = values[0].split('_')

		setattr(namespace, self.dest, values)


def custom_args(parser: ArgumentParser):

	# For perceiver
	parser.add_argument("--depth", default=4, type=int)
	parser.add_argument("--num_latents", default=512, type=int)
	parser.add_argument("--latent_dim", default=512, type=int)
	parser.add_argument("--cross_heads", default=1, type=int)
	parser.add_argument("--latent_heads", default=8, type=int)
	parser.add_argument("--cross_dim_head", default=64, type=int)
	parser.add_argument("--latent_dim_head", default=64, type=int)
	#
	parser.add_argument("--beta", default=0.5, type=float)
	parser.add_argument("--ckpt_root", default='', type=str)

	parser.add_argument("--num_outputs", default=5, type=int)
	parser.add_argument("--multi_task", choices=[0, 1], type=int, default=1)
	parser.add_argument("--target_sensitive_group", default='gender', type=str)
	parser.add_argument("--target_personality", default=None, type=int)
	parser.add_argument("--is_baseline", choices=[0, 1], type=int, default=0)
	parser.add_argument("--is_incremental", choices=[0, 1], type=int, default=0)
	parser.add_argument("--alpha", type=float, default=0.1, help='the weight of the two distribution losses')
	parser.add_argument("--use_distribution_loss", choices=[0, 1], type=int, default=0)
	parser.add_argument("--modalities", nargs='+', default=['text', 'facebody'], help='the modalities to be used', action=ParserList)
	parser.add_argument("--gamma", type=float, default=1, help='the weight of total distribution loss')

	# for biased custom dataset
	parser.add_argument("--bias_sensitive", type=str, default=None, help='create biased dataset')
	parser.add_argument("--bias_group", type=int, default=0, help='create biased dataset')
	parser.add_argument("--bias_personality", type=int, default=0, help='create biased dataset')

	# for baseline one-stage learning
	parser.add_argument("--one_stage", choices=[0, 1], type=int, default=0)



	# for MMI
	parser.add_argument('--cpc_layers', type=int, default=1,
	                    help='number of layers in CPC NCE estimator (default: 1)')
	parser.add_argument('--d_prjh', type=int, default=128,
	                    help='hidden size in projection network')
	parser.add_argument('--dropout_prj', type=float, default=0.1,
	                    help='dropout of projection layer')
	parser.add_argument('--sigma', type=float, default=0.1)
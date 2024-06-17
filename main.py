import argparse
from datetime import datetime
import json
import yaml

from loguru import logger
import numpy as np
import torch

from models import SlayNetImageOnly, SlayNetImageOnlyFSPool, SlayNet, SlayNetFSPool
from polyvore_outfits_set import OutfitSetLoader
from train import train_combined_losses, train_contrastive, train_compatibility


def main(args: argparse.Namespace):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")

    config_file_name: str = args.config_dir.split('/')[-1].split('.yaml')[0]
    logger_id: int = logger.add(f"logs/{config_file_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger.info(f"currently running configuration '{args.config_dir}'")

    if args.repeat_experiment:
        torch.manual_seed(args.torch_seed)
        if args.cuda:
            torch.cuda.manual_seed(args.torch_seed)
        logger.info(f"torch seed: {args.torch_seed}")
        np.random.seed(args.numpy_seed)
        logger.info(f"numpy seed: {args.numpy_seed}")
    else:
        logger.info(f"non-repeat experiment: torch seed set as '{args.seed}' ")
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

    fclip_embeddings = np.load(args.fclip_embeddings_path)
    with open(args.fclip_images_mapping_path, 'r') as f:
        fclip_images_mapping = f.read().split(',')
    fclip_images_mapping = {image_id: idx for idx, image_id in enumerate(fclip_images_mapping)}
    text_embeddings = None
    if args.txt_embeddings_path is not None:
        logger.info("image and text embeddings are used")
        text_embeddings = np.load(args.txt_embeddings_path)
    else:
        logger.info("only image embeddings are used")

    train_dataset = OutfitSetLoader(
        args, 'train',
        item_embeddings=fclip_embeddings,
        item_embeddings_index_mapping=fclip_images_mapping,
        item_text_embeddings=text_embeddings
    )

    valid_dataset = OutfitSetLoader(
        args, 'valid',
        item_embeddings=fclip_embeddings,
        item_embeddings_index_mapping=fclip_images_mapping,
        item_text_embeddings=text_embeddings
    )

    if args.txt_embeddings_path is None:
        if args.set_pooling_type == 'FSPool':
            model = SlayNetImageOnlyFSPool(args).to(device)
        else:
            model = SlayNetImageOnly(args).to(device)

    else:
        if args.set_pooling_type == 'FSPool':
            model = SlayNetFSPool(args).to(device)
        else:
            model = SlayNet(args).to(device)

    if args.resume != '':
        logger.info(f"training will be resumed from checkpoint '{args.resume}'")
        model.load_state_dict(torch.load(args.resume))
    else:
        logger.info("training starts fresh")

    if args.learning_type == "combined" and (args.weight_comp > 0 and args.weight_contrastive > 0):
        logger.info("training block 'combined'")
        train_dataset.prepare_for_training("contrastive")
        train_dataset_comp = OutfitSetLoader(
            args, 'train',
            item_embeddings=fclip_embeddings,
            item_embeddings_index_mapping=fclip_images_mapping,
            item_text_embeddings=text_embeddings
        )
        train_dataset_comp.prepare_for_training("compatibility")
        train_combined_losses(
            args, model, train_dataset, train_dataset_comp, args.epochs,
            valid_dataset, development_test=bool(args.development_test))

    elif args.learning_type == "compatibility" or (args.weight_comp == 1 and args.weight_contrastive == 0):
        logger.info("training block 'compatibility only'")
        train_dataset.prepare_for_training("compatibility")
        train_compatibility(
            args, model, train_dataset, args.epochs,
            valid_dataset, development_test=bool(args.development_test))

    elif args.learning_type == "contrastive" or (args.weight_contrastive == 1 and args.weight_comp == 0):
        logger.info("training block 'Triplet only'")
        train_dataset.prepare_for_training("triple_loss")
        train_contrastive(
            args, model, train_dataset, args.epochs,
            valid_dataset, development_test=bool(args.development_test))

    else:
        raise Exception("Invalid argument for 'learning_type'")

    logger.info("The execution of the script has finished")
    logger.remove(logger_id)


def read_run_config(args: argparse.Namespace):
    with open(args.config_dir, 'r') as f:
        train_config = yaml.safe_load(f)

    if 'repeat' in train_config.keys():
        logger.info(f"performing repeat experiment ...")
        this_config_dir = train_config['training']['config_path']
        repeat_resume = ''
        if 'resume' in train_config['training'].keys():
            repeat_resume = train_config['training']['resume']
        args.numpy_seed = int(train_config['repeat']['numpy_seed'])
        args.torch_seed = int(train_config['repeat']['torch_seed'])
        args.repeat_experiment = True
        with open(this_config_dir, 'r') as f:
            train_config = yaml.safe_load(f)
        if repeat_resume != '':
            train_config['training']['resume'] = repeat_resume
            logger.info(f"the following model resume path is overwritten into source train config: '{repeat_resume}'")

    # data
    args.polyvore_split = train_config['data']['polyvore_split']
    args.item_metadata_path = train_config['data']['item_metadata_path']

    args.fclip_embeddings_path = train_config['data']['fclip_embeddings_path']
    args.fclip_images_mapping_path = train_config['data']['fclip_images_mapping_path']
    args.txt_embeddings_path = train_config['data'].get('txt_embeddings_path', None)
    args.item_types_path = train_config['data']['item_types_path']

    args.data_root_dir = train_config['data']['data_root_dir']
    args.max_set_len = int(train_config['data']['max_set_len'])

    # training
    args.learning_type = train_config['training']['learning_type']
    args.negative_sample_size = int(train_config['training']['negative_sample_size'])
    args.lr = float(train_config['training']['lr'])
    args.finegrain_sampling = bool(int(train_config['training'].get('finegrain_sampling', None)))
    args.resume = train_config['training'].get('resume', '')

    # model
    args.dim_embed_img = int(train_config['model']['dim_embed_img'])
    args.dim_embed_txt = int(train_config['model'].get('dim_embed_txt', 0))
    args.set_encoder_type = train_config['model']['set_encoder_type']
    args.set_pooling_type = train_config['model']['set_pooling_type']
    args.csn_num_conditions = int(train_config['model']['csn_num_conditions'])

    # triplet loss
    args.triplet_loss_margin = float(train_config['triplet_loss']['loss_margin'])
    args.triplet_negative_aggregate = train_config['triplet_loss']['negative_aggregate']
    args.triplet_add_minimum = bool(int(train_config['triplet_loss'].get('add_minimum', None)))

    # combined loss
    args.weight_contrastive = float(train_config['combined_loss']['weight_contrastive'])
    args.weight_comp = float(train_config['combined_loss']['weight_comp'])

    # set pooling
    args.set_hidden_dim = train_config['set_pooling']['set_hidden_dim']
    args.set_reference_points_count = int(train_config['set_pooling']['set_reference_points_count'])

    assert args.learning_type in ["combined", "contrastive", "compatibility"]
    return True


def multi_configs_main(args: argparse.Namespace):
    assert (args.config_dir is not None) or (args.config_list_dir is not None)
    config_list = []
    if args.config_dir is not None:
        config_list.append(args.config_dir)

    if args.config_list_dir is not None:
        with open(args.config_list_dir, 'r') as f:
            config_list_raw = json.load(f)["config_list"]
        for i in config_list_raw:
            if i not in config_list:
                config_list.append(i)

    for each_config in config_list:
        args.config_dir = each_config
        read_run_config(args)
        main(args)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fashion_recco')
    parser.add_argument('--config_dir', type=str,
                        help='the configuration file for the training')
    parser.add_argument('--config_list_dir', type=str,
                        help='json file containing a list of train_config to be run')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--dataloader_workers', type=int, default=4,
                        help='the number of multi-processes for torch dataloader')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='force to use CPU even GPU is detected '
                             '(GPU (hence, CUDA) is used by default when it is detected')
    parser.add_argument('--checkpoint_dir', default='checkpoints', type=str,
                        help='directory of the checkpoint')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--development_test', type=int, default=0, metavar='N',
                        help='to identify if the run is for development testing (1: yes, 0: no)')
    parser.add_argument('--comp_epochs_done', type=int,
                        help='the number of epochs done for compatibility task in the last checkpoint')
    parser.add_argument('--triple_epochs_done', type=int,
                        help='the number of epochs for triple loss task in the last checkpoint')
    args = parser.parse_args()

    multi_configs_main(args)



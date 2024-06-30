import argparse
import copy
from datetime import datetime
import json
import yaml

from loguru import logger
import numpy as np
import torch

from models import SlayNetImageOnly, SlayNetImageOnlyFSPool, SlayNet, SlayNetFSPool
from polyvore_outfits_set import OutfitSetLoader
from train import train_combined_losses, train_contrastive, train_compatibility


def execute_train_combined(args, model, train_dataset, valid_dataset,
                           fclip_embeddings, fclip_images_mapping, text_embeddings):
    logger.info("training block 'combined'")
    train_dataset.prepare_for_training("contrastive")
    train_dataset_comp = OutfitSetLoader(
        args, 'train',
        item_embeddings=fclip_embeddings,
        item_embeddings_index_mapping=fclip_images_mapping,
        item_text_embeddings=text_embeddings
    )
    train_dataset_comp.prepare_for_training("compatibility")
    train_meta = train_combined_losses(
        args, model, train_dataset, train_dataset_comp, args.epochs,
        valid_dataset, development_test=bool(args.development_test))
    return train_meta


def execute_train_compatibility(args, model, train_dataset, valid_dataset):
    logger.info("training block 'compatibility only'")
    train_dataset.prepare_for_training("compatibility")
    train_meta = train_compatibility(
        args, model, train_dataset, args.epochs,
        valid_dataset, development_test=bool(args.development_test))
    return train_meta


def execute_train_contrastive(args, model, train_dataset, valid_dataset):
    logger.info("training block 'contrastive only'")
    train_dataset.prepare_for_training("contrastive")
    train_meta = train_contrastive(
        args, model, train_dataset, args.epochs,
        valid_dataset, development_test=bool(args.development_test))
    return train_meta


def main(args: argparse.Namespace):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")

    config_file_name: str = (args.config_dir.split('/')[-1].split('.yaml')[0]
                             + f"_sp{args.torch_seed}sn{args.numpy_seed}")
    logger_id: int = logger.add(f"logs/{config_file_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger.info(f"currently running configuration '{args.config_dir}'")

    torch.manual_seed(args.torch_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.torch_seed)
    logger.info(f"torch seed: {args.torch_seed}")
    np.random.seed(args.numpy_seed)
    logger.info(f"numpy seed: {args.numpy_seed}")

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
        execute_train_combined(args, model, train_dataset, valid_dataset,
                               fclip_embeddings, fclip_images_mapping, text_embeddings)

    elif args.learning_type == "compatibility" or (args.weight_comp == 1 and args.weight_contrastive == 0):
        execute_train_compatibility(args, model, train_dataset, valid_dataset)

    elif args.learning_type == "contrastive" or (args.weight_contrastive == 1 and args.weight_comp == 0):
        execute_train_contrastive(args, model, train_dataset, valid_dataset)

    elif args.learning_type == "curriculum_main":
        logger.info("training block 'curriculum - main'")

        # checking mandatory columns in the config for each phase
        for each in args.curriculum_config_overwrite:
            assert each.get('this_phase_learning_type', None) in ["combined", "compatibility", "contrastive"]

            assert int(each.get('this_phase_epochs', None)) is not None
            assert bool(int(each.get('finegrain_sampling', None))) is not None
            assert float(each.get('weight_contrastive', None)) is not None
            assert float(each.get('weight_comp', None)) is not None
            assert bool(int(each.get('contrastive_add_minimum', None))) is not None

        # this variable captures main information from previous curriculum phase
        previous_phase_result = None

        for phase_index, each_phase_config in enumerate(args.curriculum_config_overwrite):
            args.epochs = int(each_phase_config['this_phase_epochs'])
            args.finegrain_sampling = bool(int(each_phase_config.get('finegrain_sampling', None)))
            args.weight_contrastive = float(each_phase_config.get('weight_contrastive', None))
            args.weight_comp = float(each_phase_config.get('weight_comp', None))
            args.contrastive_add_minimum = bool(int(each_phase_config.get('contrastive_add_minimum', None)))
            logger.info(f"curriculum phase {phase_index + 1} | "
                        f"learning type: '{each_phase_config.get('this_phase_learning_type')}' | "
                        f"training epochs: {args.epochs}")

            # train dataset modification based on the config of current phase
            train_dataset.set_sampling_strategy('fine-grained' if args.finegrain_sampling else 'general')

            # trigger the training for the curriculum phase
            logger.info(f"previous_phase_result: {previous_phase_result}")
            if previous_phase_result is not None:
                logger.info(f"training will be resumed from checkpoint '{previous_phase_result['checkpoint_path']}'")
                model.load_state_dict(torch.load(previous_phase_result['checkpoint_path']))

            if each_phase_config.get('this_phase_learning_type', None) == "combined":
                previous_phase_result = execute_train_combined(args, model, train_dataset, valid_dataset,
                                                               fclip_embeddings, fclip_images_mapping, text_embeddings)
            elif each_phase_config.get('this_phase_learning_type', None) == "compatibility":
                previous_phase_result = execute_train_compatibility(args, model, train_dataset, valid_dataset)
            elif each_phase_config.get('this_phase_learning_type', None) == "contrastive":
                previous_phase_result = execute_train_contrastive(args, model, train_dataset, valid_dataset)
            else:
                raise Exception("invalid value for 'this_phase_learning_type'")
    else:
        raise Exception("Invalid argument for 'learning_type'")

    logger.info("The execution of the script has finished")
    logger.remove(logger_id)


def read_run_config(args: argparse.Namespace):
    with open(args.config_dir, 'r') as f:
        train_config = yaml.safe_load(f)

    # data
    args.polyvore_split = train_config['data']['polyvore_split']
    args.item_metadata_path = train_config['data']['item_metadata_path']

    args.contrastive_learning_data_path = train_config['data']['contrastive_learning_data_path']
    args.fclip_embeddings_path = train_config['data']['fclip_embeddings_path']
    args.fclip_images_mapping_path = train_config['data']['fclip_images_mapping_path']
    args.txt_embeddings_path = train_config['data'].get('txt_embeddings_path', None)
    args.item_types_path = train_config['data']['item_types_path']

    args.data_root_dir = train_config['data']['data_root_dir']
    args.max_outfit_length = int(train_config['data']['max_outfit_length'])

    # training
    args.learning_type = train_config['training']['learning_type']
    args.negative_sample_size = int(train_config['training']['negative_sample_size'])
    args.lr = float(train_config['training']['lr'])
    args.resume = train_config['training'].get('resume', '')
    args.finegrain_sampling = bool(int(train_config['training'].get('finegrain_sampling', 0)))
    if args.learning_type == 'curriculum_main':
        args.curriculum_config_overwrite = []
        args.curriculum_config_overwrite.append(train_config['training']['curriculum_phase_1'])
        args.curriculum_config_overwrite.append(train_config['training']['curriculum_phase_2'])

    # random seed
    args.torch_seed = train_config['random_seed']['torch']
    args.numpy_seed = train_config['random_seed']['numpy']

    # model
    args.dim_embed_img = int(train_config['model']['dim_embed_img'])
    args.dim_embed_txt = int(train_config['model'].get('dim_embed_txt', 0))
    args.set_encoder_type = train_config['model']['set_encoder_type']
    args.set_pooling_type = train_config['model']['set_pooling_type']
    args.csa_num_conditions = int(train_config['model']['csa_num_conditions'])

    # contrastive loss
    args.contrastive_loss_margin = float(train_config['contrastive_loss']['loss_margin'])
    args.contrastive_negative_aggregate = train_config['contrastive_loss']['negative_aggregate']
    args.contrastive_add_minimum = bool(int(train_config['contrastive_loss'].get('add_minimum', 0)))

    # combined loss
    args.weight_contrastive = float(train_config['combined_loss']['weight_contrastive'])
    args.weight_comp = float(train_config['combined_loss']['weight_comp'])

    # set pooling
    args.set_hidden_dim = train_config['set_pooling']['set_hidden_dim']
    args.set_reference_points_count = int(train_config['set_pooling']['set_reference_points_count'])

    assert args.learning_type in ["combined", "contrastive", "compatibility", "curriculum_main"]
    return True


def multi_configs_main(args: argparse.Namespace):
    assert ((args.config_dir is not None) or (args.config_list_dir is not None) or
            (args.repeat_config_list_dir is not None))

    if args.repeat_config_list_dir is None:
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

    else:
        with open(args.repeat_config_list_dir, 'r') as f:
            this_raw_config = json.load(f)
            config_list_raw = this_raw_config["config_list"]
            seed_list_raw = this_raw_config["seed_list"]
        for i in config_list_raw:
            for this_torch_seed, this_numpy_seed in seed_list_raw:
                args.config_dir = i
                read_run_config(args)
                args.torch_seed = this_torch_seed
                args.numpy_seed = this_numpy_seed
                main(args)
        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fashion_recco')
    parser.add_argument('--config_dir', type=str,
                        help='the configuration file for the training')
    parser.add_argument('--config_list_dir', type=str,
                        help='json file containing a list of train_config to be run')
    parser.add_argument('--repeat_config_list_dir', type=str,
                        help='json file containing a list of train_config to be run '
                             'and the seed list for repeat training')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--dataloader_workers', type=int, default=4,
                        help='the number of multi-processes for torch dataloader')
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
    parser.add_argument('--pytorch_compile', type=int, default=1,
                        help='the flag to enable compilation of pytorch code to accelerate code run')
    args = parser.parse_args()

    multi_configs_main(args)



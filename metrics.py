import argparse
import os

from loguru import logger
import numpy as np
import torch
import torch.nn as nn
from torchmetrics import AUROC


from models import SlayNetImageOnly, SlayNetImageOnlyFSPool, SlayNet, SlayNetFSPool

from polyvore_outfits_set import fitb_set_collation, compatibility_set_collation, OutfitSetLoader, ItemEmbeddingLoader


def calculate_fitb(args, model, dataset, batch_size=128, collate_function=fitb_set_collation,
                   development_test=False):
    model.eval()

    kwargs = {'num_workers': args.dataloader_workers, 'pin_memory': True} if args.cuda else {}

    dataset.prepare_for_metrics_calculation('fitb')
    this_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if development_test else False,
        collate_fn=collate_function,
        **kwargs
    )

    input_device = torch.device("cuda:0" if args.cuda else "cpu")

    correct_answers = 0
    instances_num = 0
    logger.info('Calculating metric FITB: ')
    with torch.no_grad():
        for each_batch in iter(this_dataloader):
            q_img, q_onehot, q_img_mask, project_to_onehot, a_img, a_onehot, outfit_len, \
                q_txt, a_txt = each_batch
            q_img = q_img.to(input_device)
            q_onehot = q_onehot.to(input_device)
            q_img_mask = q_img_mask.to(input_device)
            project_to_onehot = project_to_onehot.to(input_device)
            a_img = a_img.to(input_device)
            a_onehot = a_onehot.to(input_device)
            outfit_len = outfit_len.to(input_device)
            q_txt = q_txt.to(input_device)
            a_txt = a_txt.to(input_device)

            f_q = model(
                q_img, q_onehot, project_to_onehot,
                feat_mask=q_img_mask, set_size=outfit_len, txt=q_txt)
            f_a = model.forward_negative(a_img, a_onehot, txt=a_txt)

            batch_dim = f_q.shape[0]
            embedding_dim = f_q.shape[-1]
            answers_num = f_a.shape[1]

            # calculate euclidian distance between outfit embedding and each items in answer
            pred = torch.sqrt(torch.sum(
                torch.square(f_q.unsqueeze(1).expand((batch_dim, answers_num, embedding_dim)) - f_a), dim=-1
            ))

            # the correct answer is always on index 0 for each instance
            correct_answers += torch.sum((torch.argmin(pred, dim=-1) == 0).long())
            instances_num += q_img.shape[0]

            print(instances_num, end=', ')
            if development_test:
                break
    return correct_answers / instances_num


def calculate_compatibility_auc(args, model, dataset, batch_size=128, collate_function=compatibility_set_collation,
                               development_test=False):
    auroc = AUROC(task="binary")
    model.eval()

    kwargs = {'num_workers': args.dataloader_workers, 'pin_memory': True} if args.cuda else {}

    dataset.prepare_for_metrics_calculation("compatibility")
    this_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if development_test else False,
        collate_fn=collate_function,
        **kwargs
    )
    sigmoid = nn.Sigmoid()

    input_device = torch.device("cuda:0" if args.cuda else "cpu")

    instances_num = 0
    logger.info('Calculating metric compatibility AUC: ')
    with torch.no_grad():
        for each_batch in iter(this_dataloader):
            outfit_img, outfit_onehot, outfit_img_mask, outfit_label, outfit_len, outfit_txt = each_batch
            outfit_img = outfit_img.to(input_device)
            outfit_onehot = outfit_onehot.to(input_device)
            outfit_img_mask = outfit_img_mask.to(input_device)
            outfit_label = outfit_label.to(input_device)
            outfit_len = outfit_len.to(input_device)
            outfit_txt = outfit_txt.to(input_device)

            f_compatibility = model.forward_compatibility(
                outfit_img, outfit_onehot,
                feat_mask=outfit_img_mask, set_size=outfit_len, txt=outfit_txt)
            f_compatibility = sigmoid(f_compatibility)
            auroc.update(f_compatibility, outfit_label)

            instances_num += outfit_img.shape[0]
            print(instances_num, end=', ')
            if development_test:
                break
    return auroc.compute()


def load_data_for_metrics_calculation(args: argparse.Namespace, data_split: str, metric_type: str = 'comp-fitb'):
    assert data_split in ['valid', 'test']
    assert metric_type in ['comp-fitb', 'rak']
    text_embeddings = None
    if args.txt_embeddings_path is not None:
        logger.info("image and text embeddings are used")
        text_embeddings = np.load(args.txt_embeddings_path)
    else:
        logger.info("only image embeddings are used")

    fclip_embeddings = np.load(args.fclip_embeddings_path)
    with open(args.fclip_images_mapping_path, 'r') as f:
        fclip_images_mapping = f.read().split(',')
    fclip_images_mapping = {image_id: idx for idx, image_id in enumerate(fclip_images_mapping)}
    if metric_type == 'comp-fitb':
        output_dataset = OutfitSetLoader(
            args, data_split,
            item_embeddings=fclip_embeddings,
            item_embeddings_index_mapping=fclip_images_mapping,
            item_text_embeddings=text_embeddings
        )
    else:
        output_dataset = ItemEmbeddingLoader(
            args, data_split,
            item_embeddings=fclip_embeddings,
            item_embeddings_index_mapping=fclip_images_mapping,
            item_text_embeddings=text_embeddings
        )
    return output_dataset


def load_model_for_metrics_calculation(args: argparse.Namespace, checkpoint_dir: str, ):
    assert os.path.isfile(checkpoint_dir)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")

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
    logger.info(f"model is load from checkpoint '{checkpoint_dir}' ")
    model.load_state_dict(torch.load(checkpoint_dir))
    return model


def calculate_metrics_and_save_result(args, this_model, input_dataset, output_path, development_test=False):
    this_fitb = calculate_fitb(args, this_model, input_dataset,
                               batch_size=args.batch_size, development_test=development_test)
    this_comp_auc = calculate_compatibility_auc(args, this_model, input_dataset,
                                                batch_size=args.batch_size, development_test=development_test)
    with open(output_path, 'w') as f:
        f.write(f"fitb: {this_fitb.item()}\ncompatibility AUC: {this_comp_auc.item()}")
    return True


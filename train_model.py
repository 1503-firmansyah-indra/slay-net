import argparse
import time
from contextlib import nullcontext
import sys

from lightning.pytorch.utilities.combined_loader import CombinedLoader
from loguru import logger
import torch
import torch.nn as nn
from torchmetrics import AUROC

from train_utils import CheckpointManager
from metrics import calculate_fitb, calculate_compatibility_auc
from models import FashRecco
from polyvore_outfits_set import OutfitSetLoader, triple_loss_set_collation

class SetWiseRankingLoss(torch.nn.Module):
    def __init__(self, loss_margin, negative_aggregation, num_negative_sample, device):
        super(SetWiseRankingLoss, self).__init__()
        self.num_negative_sample = num_negative_sample
        self.margin_ranking_loss = nn.MarginRankingLoss(margin=loss_margin)
        if negative_aggregation == 'mean':
            self.aggregate_negatives = torch.mean
        else:
            raise Exception('Not implemented yet')
        self.device = device

    def forward(self, anchor, positives, negatives):
        D_p = torch.sqrt(
            torch.sum(
                torch.square(anchor - positives), dim=-1
            )
        )

        D_n = torch.sqrt(
            torch.sum(
                torch.square(
                    anchor.unsqueeze(1).expand((anchor.shape[0], self.num_negative_sample, anchor.shape[-1])) - negatives
                ), dim=-1
            )
        )
        D_n = self.aggregate_negatives(D_n, dim=-1)
        return self.margin_ranking_loss(D_p, D_n, torch.full(D_p.shape, -1, device=self.device))


class SetWiseRankingLossHard(SetWiseRankingLoss):
    def __init__(self, loss_margin, negative_aggregation, num_negative_sample, device):
        super(SetWiseRankingLossHard, self).__init__(
            loss_margin, negative_aggregation, num_negative_sample, device)

    def forward(self, anchor, positives, negatives):
        D_p = torch.sqrt(
            torch.sum(
                torch.square(anchor - positives), dim=-1
            )
        )

        D_n_raw = torch.sqrt(
            torch.sum(
                torch.square(
                    anchor.unsqueeze(1).expand(
                        (anchor.shape[0], self.num_negative_sample, anchor.shape[-1])) - negatives
                ), dim=-1
            )
        )
        D_n = self.aggregate_negatives(D_n_raw, dim=-1)
        D_n_min, _ = torch.min(D_n_raw, -1)
        return self.margin_ranking_loss(D_p, D_n, torch.full(D_p.shape, -1, device=self.device)) \
            + self.margin_ranking_loss(D_p, D_n_min, torch.full(D_p.shape, -1, device=self.device))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_metrics_and_print(args, model, dataset, metric_batch, development_test, epoch_count):
    '''

    :param model:
    :param dataset:
    :param metric_batch:
    :param development_test:
    :param epoch_count:
    :return:
    '''
    this_epoch_fitb = calculate_fitb(
        args, model, dataset, batch_size=metric_batch, development_test=development_test)
    this_epoch_comp_auc = calculate_compatibility_auc(
        args, model, dataset, batch_size=metric_batch, development_test=development_test)
    logger.info(f"Epoch: {epoch_count} | FITB accuracy: {round(this_epoch_fitb.item(), 5)} | "
          f"Compatibility accuracy: {round(this_epoch_comp_auc.item(), 5)}")
    return True


def train_combined_losses(args: argparse.Namespace, model: FashRecco,
                          contrastive_dataset: OutfitSetLoader, comp_dataset: OutfitSetLoader,
                          epoch_num: int, valid_dataset: OutfitSetLoader,
                          valid_metric_batch: int = 128, development_test: bool = False, print_interval: int = 200):
    # commented because compiling model caused RuntimeError: Triton Error [CUDA]: misaligned address
    # if sys.platform != 'win32':
    #     model = torch.compile(model)

    # general hyper params
    learning_rate = args.lr
    num_negative_sample = args.negative_sample_size

    # for triplet loss
    triplet_loss_margin = args.triplet_loss_margin
    logger.info(f"triplet loss margin: {triplet_loss_margin}")
    triplet_negative_aggregate = args.triplet_negative_aggregate

    # for smooth AP
    smoothap_sigmoid_temperature = args.smoothap_sigmoid_temperature
    smoothap_similarity_measure = args.smoothap_similarity_measure

    # for loss combination
    weight_triplet = args.weight_triplet
    weight_smoothap = args.weight_smoothap
    weight_comp = args.weight_comp
    logger.info(f"Loss weights: triplet = {weight_triplet} | smooth AP = {weight_smoothap} | compatibility = {weight_comp}")

    checkpoint_dir = args.checkpoint_dir

    kwargs = {'num_workers': args.dataloader_workers, 'pin_memory': True} if args.cuda else {}

    dataloader_triplet = torch.utils.data.DataLoader(
        contrastive_dataset,
        batch_size=args.batch_size, shuffle=True,
        #collate_fn=triple_loss_set_collation,
        **kwargs
    )
    dataloader_comp = torch.utils.data.DataLoader(
        comp_dataset,
        batch_size=args.batch_size, shuffle=True,
        #collate_fn=compatibility_set_collation,
        **kwargs
    )

    input_device = torch.device("cuda:0" if args.cuda else "cpu")

    bce_loss_calculator = nn.BCEWithLogitsLoss()
    if args.triplet_add_minimum:
        logger.info("Triplet loss variant: aggregate and minimum")
        triplet_loss_calculator = SetWiseRankingLossHard(
            triplet_loss_margin, triplet_negative_aggregate, num_negative_sample, input_device)
    else:
        logger.info("Triplet loss variant: aggregate only")
        triplet_loss_calculator = SetWiseRankingLoss(
            triplet_loss_margin, triplet_negative_aggregate, num_negative_sample, input_device)
    if sys.platform != 'win32':
        triplet_loss_calculator = torch.compile(triplet_loss_calculator)
    smoothap_loss_calculator = SmoothAP(
        input_device, anneal=smoothap_sigmoid_temperature, similarity_measure=smoothap_similarity_measure)
    if sys.platform != 'win32':
        smoothap_loss_calculator = torch.compile(smoothap_loss_calculator)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    if args.cuda:
        scaler = torch.cuda.amp.GradScaler()

    # checkpoint name modifier
    this_checkpoint_manager = CheckpointManager(args)

    start_time = time.time()
    logger.info('Combined Losses training has started...')
    logger.info(f"The number of learnable parameters of the model: {count_parameters(model)}")

    # for compatibility (Binary Cross Entropy loss)
    auroc = AUROC(task="binary")
    sigmoid = nn.Sigmoid()

    iterables = {
        'triplet': dataloader_triplet,
        'comp': dataloader_comp
    }
    combined_loader = CombinedLoader(iterables, 'max_size_cycle')

    model.train()
    instance_count = 0
    for e in range(epoch_num):
        for batch, iter_count, _ in combined_loader:
            pos_img, pos_onehot, outfit_img, outfit_onehot, out_img_mask, \
                neg_img, neg_onehot, triplet_outfit_len, \
                pos_txt, outfit_txt, neg_txt = batch['triplet']
            pos_img = pos_img.to(input_device)
            pos_onehot = pos_onehot.to(input_device)
            outfit_img = outfit_img.to(input_device)
            outfit_onehot = outfit_onehot.to(input_device)
            out_img_mask = out_img_mask.to(input_device)
            neg_img = neg_img.to(input_device)
            neg_onehot = neg_onehot.to(input_device)
            triplet_outfit_len = triplet_outfit_len.to(input_device)
            pos_txt = pos_txt.to(input_device)
            outfit_txt = outfit_txt.to(input_device)
            neg_txt = neg_txt.to(input_device)

            c_outfit_img, c_outfit_onehot, c_outfit_img_mask, c_outfit_label, c_outfit_len, c_outfit_txt = batch['comp']
            c_outfit_img = c_outfit_img.to(input_device)
            c_outfit_onehot = c_outfit_onehot.to(input_device)
            c_outfit_img_mask = c_outfit_img_mask.to(input_device)
            c_outfit_label = c_outfit_label.to(input_device)
            c_outfit_len = c_outfit_len.to(input_device)
            c_outfit_txt = c_outfit_txt.to(input_device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast() if args.cuda else nullcontext():
                # Compatibility (Binary Cross Entropy)
                f_compatibility = model.forward_compatibility(
                    c_outfit_img, c_outfit_onehot,
                    feat_mask=c_outfit_img_mask, set_size=c_outfit_len, txt=c_outfit_txt
                )
                f_compatibility = f_compatibility.squeeze(-1)

                bce_loss = bce_loss_calculator(f_compatibility, c_outfit_label.float())
                auroc.update(sigmoid(f_compatibility), c_outfit_label)

                loss = weight_comp*bce_loss

                # calculate representation for anchor, positive and negative
                f_o = model(
                    outfit_img, outfit_onehot, pos_onehot,
                    feat_mask=out_img_mask, set_size=triplet_outfit_len, txt=outfit_txt)
                f_p = model.forward_positive(pos_img, pos_onehot, txt=pos_txt)
                f_n = model.forward_negative(neg_img, neg_onehot, txt=neg_txt)

                # Triplet loss
                if weight_triplet > 0:
                    triplet_loss = triplet_loss_calculator(f_o, f_p, f_n)
                    loss += weight_triplet*triplet_loss

                # Smooth AP
                if weight_smoothap > 0:
                    candidates = torch.cat((f_p.unsqueeze(1), f_n), dim=1)
                    smoothap_loss = smoothap_loss_calculator(f_o, candidates)
                    loss += weight_smoothap*smoothap_loss

                if not torch.all(torch.isfinite(loss)):
                    logger.info("Loss becomes infinite, stopping the training early")
                    return True

            if args.cuda:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            instance_count += outfit_img.shape[0]
            if iter_count % print_interval == 0:
                print_message = f"Epoch: {e} | iteration {iter_count} | loss {round(loss.item(), 5)}" \
                                f" | BCE loss {round(bce_loss.item(), 5)}"
                if weight_triplet > 0:
                    print_message += f" | Triplet loss {round(triplet_loss.item(), 5)}"
                if weight_smoothap > 0:
                    print_message += f" | Smooth AP loss {round(smoothap_loss.item(), 5)}"
                logger.info(print_message)
            if development_test:
                break
        this_train_auc = auroc.compute().item()
        logger.info(f"Epoch: {e} | train AUC {round(this_train_auc, 5)} ")
        auroc.reset()

        # saving the model checkpoint
        checkpoint_path = this_checkpoint_manager.generate_name(
            'combined', checkpoint_dir, e, instance_count
        )
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"A checkpoint has just saved in '{checkpoint_path}'")

        # calculating metrics using validation dataset
        calculate_metrics_and_print(args, model, valid_dataset, valid_metric_batch, development_test, e)

        if development_test:
            break
    end_time = time.time()
    time_taken = end_time - start_time
    logger.info(f"The training took: {round(time_taken, 1)} seconds")
    return True


def train_contrastive(args: argparse.Namespace, model: FashRecco, dataset: OutfitSetLoader, epoch_num: int,
                      valid_dataset: OutfitSetLoader,
                      valid_metric_batch: int = 128, development_test: bool = False, print_interval: int = 200):
    # general hyper params
    learning_rate = args.lr
    num_negative_sample = args.negative_sample_size

    # for triplet loss
    triplet_loss_margin = args.triplet_loss_margin
    logger.info(f"triplet loss margin: {triplet_loss_margin}")
    triplet_negative_aggregate = args.triplet_negative_aggregate

    input_device = torch.device("cuda:0" if args.cuda else "cpu")
    checkpoint_dir = args.checkpoint_dir

    kwargs = {'num_workers': args.dataloader_workers, 'pin_memory': True} if args.cuda else {}

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size, shuffle=True,
        **kwargs
    )

    if args.triplet_add_minimum:
        logger.info("Triplet loss variant: aggregate and minimum")
        triplet_loss_calculator = SetWiseRankingLossHard(
            triplet_loss_margin, triplet_negative_aggregate, num_negative_sample, input_device)
    else:
        logger.info("Triplet loss variant: aggregate only")
        triplet_loss_calculator = SetWiseRankingLoss(
            triplet_loss_margin, triplet_negative_aggregate, num_negative_sample, input_device)
    if sys.platform != 'win32':
        triplet_loss_calculator = torch.compile(triplet_loss_calculator)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    if args.cuda:
        scaler = torch.cuda.amp.GradScaler()

    # checkpoint name modifier
    this_checkpoint_manager = CheckpointManager(args)

    start_time = time.time()
    logger.info('Triple Loss training has started...')
    logger.info(f"The number of learnable parameters of the model: {count_parameters(model)}")

    model.train()
    instance_count = 0
    for e in range(epoch_num):
        for iter_count, each_batch in enumerate(dataloader):
            pos_img, pos_onehot, outfit_img, outfit_onehot, out_img_mask, \
                neg_img, neg_onehot, triplet_outfit_len, \
                pos_txt, outfit_txt, neg_txt = each_batch
            pos_img = pos_img.to(input_device)
            pos_onehot = pos_onehot.to(input_device)
            outfit_img = outfit_img.to(input_device)
            outfit_onehot = outfit_onehot.to(input_device)
            out_img_mask = out_img_mask.to(input_device)
            neg_img = neg_img.to(input_device)
            neg_onehot = neg_onehot.to(input_device)
            triplet_outfit_len = triplet_outfit_len.to(input_device)
            pos_txt = pos_txt.to(input_device)
            outfit_txt = outfit_txt.to(input_device)
            neg_txt = neg_txt.to(input_device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast() if args.cuda else nullcontext():
                f_o = model(
                    outfit_img, outfit_onehot, pos_onehot,
                    feat_mask=out_img_mask, set_size=triplet_outfit_len, txt=outfit_txt)
                f_p = model.forward_positive(pos_img, pos_onehot, txt=pos_txt)
                f_n = model.forward_negative(neg_img, neg_onehot, txt=neg_txt)

                loss = triplet_loss_calculator(f_o, f_p, f_n)
                if not torch.all(torch.isfinite(loss)):
                    logger.info("Loss becomes infinite, stopping the training early")
                    return True

            if args.cuda:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            instance_count += pos_img.shape[0]

            if iter_count % print_interval == 0:
                logger.info(f"Epoch: {e} | iteration {iter_count} | loss {round(loss.item(), 5)}")
            if development_test:
                break
        # saving the model checkpoint
        checkpoint_path = this_checkpoint_manager.generate_name(
            'triple_loss', checkpoint_dir, e, instance_count
        )
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"A checkpoint has just saved in '{checkpoint_path}'")

        # calculating metrics using validation dataset
        calculate_metrics_and_print(args, model, valid_dataset, valid_metric_batch, development_test, e)

        if development_test:
            break
    end_time = time.time()
    time_taken = end_time - start_time
    logger.info(f"The training took: {round(time_taken, 1)} seconds")
    return True


def train_compatibility(args: argparse.Namespace, model: FashRecco, dataset: OutfitSetLoader, epoch_num: int,
                        valid_dataset: OutfitSetLoader,
                        valid_metric_batch: int = 128, development_test: bool = False, print_interval: int = 200):
    learning_rate = args.lr
    checkpoint_dir = args.checkpoint_dir

    kwargs = {'num_workers': args.dataloader_workers, 'pin_memory': True} if args.cuda else {}
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size, shuffle=True,
        #collate_fn=compatibility_set_collation,
        **kwargs
    )

    loss_calculator = nn.BCEWithLogitsLoss()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    input_device = torch.device("cuda:0" if args.cuda else "cpu")

    if args.cuda:
        scaler = torch.cuda.amp.GradScaler()

    # checkpoint name modifier
    this_checkpoint_manager = CheckpointManager(args)

    start_time = time.time()
    logger.info('Compatibility training has started...')
    logger.info(f"The number of learnable parameters of the model: {count_parameters(model)}")

    auroc = AUROC(task="binary")
    sigmoid = nn.Sigmoid()

    model.train()
    instance_count = 0
    for e in range(epoch_num):
        for iter_count, each_batch in enumerate(dataloader):
            c_outfit_img, c_outfit_onehot, c_outfit_img_mask, c_outfit_label, c_outfit_len, c_outfit_txt = each_batch
            c_outfit_img = c_outfit_img.to(input_device)
            c_outfit_onehot = c_outfit_onehot.to(input_device)
            c_outfit_img_mask = c_outfit_img_mask.to(input_device)
            c_outfit_label = c_outfit_label.to(input_device)
            c_outfit_len = c_outfit_len.to(input_device)
            c_outfit_txt = c_outfit_txt.to(input_device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast() if args.cuda else nullcontext():
                f_compatibility = model.forward_compatibility(
                    c_outfit_img, c_outfit_onehot,
                    feat_mask=c_outfit_img_mask, set_size=c_outfit_len, txt=c_outfit_txt
                )
                f_compatibility = f_compatibility.squeeze(-1)

                loss = loss_calculator(f_compatibility, c_outfit_label.float())
                if not torch.all(torch.isfinite(loss)):
                    logger.info("Loss becomes infinite, stopping the training early")
                    return True
                auroc.update(sigmoid(f_compatibility), c_outfit_label)

            if args.cuda:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            instance_count += c_outfit_img.shape[0]
            if iter_count % print_interval == 0:
                logger.info(f"Epoch: {e} | iteration {iter_count} | loss {round(loss.item(), 5)}")
            if development_test:
                break
        this_train_auc = auroc.compute().item()
        logger.info(f"Epoch: {e} | train AUC {round(this_train_auc, 5)} ")
        auroc.reset()

        # saving the model checkpoint
        checkpoint_path = this_checkpoint_manager.generate_name(
            'compatibility', checkpoint_dir, e, instance_count
        )
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"A checkpoint has just saved in '{checkpoint_path}'")

        # calculating metrics using validation dataset
        calculate_metrics_and_print(args, model, valid_dataset, valid_metric_batch, development_test, e)

        if development_test:
            break
    end_time = time.time()
    time_taken = end_time - start_time
    logger.info(f"The training took: {round(time_taken, 1)} seconds")
    return True


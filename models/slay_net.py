# some parts from this script are taken from https://github.com/juho-lee/set_transformer/blob/master/modules.py

import argparse
import math

from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pswe import PSWE
from .fspool import FSPool


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class SetEncoderSAB(nn.Module):
    def __init__(self, dim_input: int, dim_output: int,
                 dim_hidden: int = 256, num_heads: int = 4, ln: bool = False):
        super(SetEncoderSAB, self).__init__()
        self.encoder = nn.Sequential(
            SAB(dim_input, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_output, num_heads, ln=ln)
        )

    def forward(self, x):
        return self.encoder(x)


class SetEncoderSABLarge(nn.Module):
    def __init__(self, dim_input: int, dim_output: int,
                 dim_hidden: int = 256, num_heads: int = 16, ln: bool = False):
        super(SetEncoderSABLarge, self).__init__()
        self.encoder = nn.Sequential(
            SAB(dim_input, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_output, num_heads, ln=ln)
        )

    def forward(self, x):
        return self.encoder(x)


class PoolingPMA(nn.Module):
    def __init__(self, dim_input, dim_output, num_outputs=1,
                 num_heads=4, ln=False):
        super(PoolingPMA, self).__init__()
        self.pooling = nn.Sequential(
            PMA(dim_input, num_heads, num_outputs, ln=ln),
            nn.Linear(dim_input, dim_output)
        )

    def forward(self, x):
        return self.pooling(x)


class CSA(nn.Module):
    def __init__(self, device: torch.device, embedding_size: int, number_of_types: int,
                 num_conditions: int = 5):
        super(CSA, self).__init__()

        self.num_conditions = num_conditions
        self.num_category = number_of_types  # the number of item type
        self.embedding_size = embedding_size

        self.condition_index = torch.arange(self.num_conditions).to(device)

        self.cate_net = nn.Sequential(
            nn.Linear(self.num_category, self.num_conditions),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_conditions, self.num_conditions),
            nn.Softmax(dim=1)
        )

        self.masks = nn.Embedding(self.num_conditions, self.embedding_size)
        self.masks.weight.data.normal_(0.9, 0.7)

    def forward(self, X, target_onehot):
        '''
            X.shape = batch, embedding_dim
        '''
        batch_dim, embedding_dim = X.shape
        assert embedding_dim == self.embedding_size
        X = X.expand((self.num_conditions, batch_dim, embedding_dim))
        X = torch.transpose(X, 0, 1)  # reshape such that batch, num_conditions, embedding_dim
        X = self.masks(self.condition_index) * X

        condition_weight = self.cate_net(target_onehot).unsqueeze(-1)
        condition_batch, _, _ = condition_weight.shape
        assert condition_batch == batch_dim
        condition_weight = condition_weight.expand((
            condition_batch, self.num_conditions, self.embedding_size))

        X = condition_weight * X
        return torch.sum(X, dim=1)


class SlayNetImageOnly(nn.Module):
    def __init__(
            self,
            args: argparse.Namespace,
            number_of_types: int = 11,
            raw_img_embedding_dim: int = 512
    ):
        super(SlayNetImageOnly, self).__init__()

        self.image_embedding_size = args.dim_embed_img
        logger.info(f"image embedding size: {self.image_embedding_size}")
        self.text_embedding_size = args.dim_embed_txt
        logger.info(f"text embedding size: {self.text_embedding_size}")
        self.embedding_size = self.image_embedding_size + self.text_embedding_size
        logger.info(f"overall embedding size: {self.embedding_size}")
        self.neg_sample_size = args.negative_sample_size
        device = torch.device("cuda:0" if args.cuda else "cpu")

        # this corresponds to the number of types in file "item_types.json"
        self.type_onehot_dim = number_of_types

        # this is "Image Encoder Block"
        self.image_encoder_block = nn.Sequential(
            nn.Linear(raw_img_embedding_dim, 256, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128, bias=False),
            nn.ReLU(),
            nn.Linear(128, self.image_embedding_size, bias=False),
        )

        self.text_encoder_block = None

        assert args.set_encoder_type in ['SABS', 'MLP', 'SABL']
        self.set_hidden_dim = args.set_hidden_dim
        if args.set_encoder_type == 'SABS':
            logger.info('set encoder: SABS')
            self.set_encoder = SetEncoderSAB(
                self.embedding_size + self.type_onehot_dim,
                self.set_hidden_dim)
        elif args.set_encoder_type == 'SABL':
            logger.info('set encoder: SABL')
            self.set_encoder = SetEncoderSABLarge(
                self.embedding_size + self.type_onehot_dim,
                self.set_hidden_dim
            )
        elif args.set_encoder_type == 'MLP':
            logger.info('set encoder: MLP')
            self.set_encoder = nn.Sequential(
                nn.Linear(self.embedding_size + self.type_onehot_dim, 128),
                nn.ReLU(),
                nn.Linear(128, self.set_hidden_dim)
            )
        else:
            raise Exception("Invalid argument 'set_encoder_type'")

        assert args.set_pooling_type in ['PMA', 'PSWE', 'FSPool']
        if args.set_pooling_type == 'PMA':
            logger.info('set pooling: PMA')
            self.set_pooling = PoolingPMA(self.set_hidden_dim, self.embedding_size)
        elif args.set_pooling_type == 'PSWE':
            logger.info('set pooling: PSWE')
            self.set_reference_points_count = args.set_reference_points_count
            self.set_pooling = PSWE(
                self.set_hidden_dim,
                self.set_reference_points_count,
                self.embedding_size)
        elif args.set_pooling_type == 'FSPool':
            logger.info('set pooling: FSPool')
            self.set_reference_points_count = args.set_reference_points_count
            self.set_pooling = FSPool(self.set_hidden_dim, self.set_reference_points_count, )
        else:
            raise Exception("Invalid argument 'set_pooling_type'")

        self.set_layer = nn.Sequential(
            self.set_encoder,
            self.set_pooling
        )

        # this is "Binary Classification Head"
        self.compatibility_layer = nn.Sequential(
            nn.Linear(self.embedding_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        # this is "Contrastive Learning Head"
        logger.info(f"Number of CSA subspace(s): {args.csa_num_conditions}")
        self.csa_layer = CSA(
            device, self.embedding_size, self.type_onehot_dim,
            num_conditions=args.csa_num_conditions)

    def forward(self, embedding, onehot, project_to_onehot, feat_mask=None, set_size=None, txt=None):
        dim_batch, dim_set, embedding_dim = embedding.shape
        # merging batch and set size so that the tensor can be passed to backbone
        embedding = embedding.reshape(dim_batch * dim_set, embedding_dim)
        feat = self.image_encoder_block(embedding)
        feat = feat.reshape(dim_batch, dim_set, feat.shape[-1])
        feat = torch.cat([feat, onehot], dim=-1)

        feat = self.set_layer(feat)
        feat = feat.squeeze(1)

        feat = self.csa_layer(feat, project_to_onehot)
        return feat

    def forward_positive(self, embedding, onehot, txt=None):
        feat = self.image_encoder_block(embedding)
        feat = self.csa_layer(feat, onehot)
        return feat

    def forward_negative(self, embedding, onehot, txt=None):
        dim_batch, dim_set, embedding_dim = embedding.shape
        # merging batch and set size so that the tensor can be passed to backbone
        embedding = embedding.reshape(dim_batch * dim_set, embedding_dim)
        feat = self.image_encoder_block(embedding)
        onehot = onehot.reshape(
            onehot.shape[0] * onehot.shape[1],
            onehot.shape[2]
        )
        feat = self.csn_layer(feat, onehot)
        feat = feat.reshape(dim_batch, dim_set, feat.shape[-1])
        return feat

    def forward_compatibility(self, embedding, onehot, feat_mask=None, set_size=None, txt=None):
        dim_batch, dim_set, embedding_dim = embedding.shape
        # merging batch and set size so that the tensor can be passed to backbone
        embedding = embedding.reshape(dim_batch * dim_set, embedding_dim)
        feat = self.image_encoder_block(embedding)
        feat = feat.reshape(dim_batch, dim_set, feat.shape[-1])
        feat = torch.cat([feat, onehot], dim=-1)

        feat = self.set_layer(feat)
        feat = feat.squeeze(1)

        feat = self.compatibility_layer(feat)
        return feat


class SlayNetImageOnlyFSPool(SlayNetImageOnly):
    def __init__(
            self,
            args: argparse.Namespace,
            number_of_types: int = 11,
            raw_img_embedding_dim: int = 512
    ):
        assert args.set_pooling_type == 'FSPool'
        super(SlayNetImageOnlyFSPool, self).__init__(
            args,
            number_of_types=number_of_types,
            raw_img_embedding_dim=raw_img_embedding_dim
        )
        self.fspool_linear_layer = nn.Linear(args.set_hidden_dim, args.dim_embed_img)

    def forward(self, embedding, onehot, project_to_onehot, feat_mask=None, set_size=None, txt=None):
        dim_batch, dim_set, embedding_dim = embedding.shape
        # merging batch and set size so that the tensor can be passed to backbone
        embedding = embedding.reshape(dim_batch * dim_set, embedding_dim)
        feat = self.image_encoder_block(embedding)
        feat = feat.reshape(dim_batch, dim_set, feat.shape[-1])
        feat = torch.cat([feat, onehot], dim=-1)

        feat = self.set_encoder(feat)
        feat = self.set_pooling(feat, set_size)
        feat = self.fspool_linear_layer(feat)

        feat = self.csn_layer(feat, project_to_onehot)
        return feat

    def forward_compatibility(self, embedding, onehot, feat_mask=None, set_size=None, txt=None):
        dim_batch, dim_set, embedding_dim = embedding.shape
        # merging batch and set size so that the tensor can be passed to backbone
        embedding = embedding.reshape(dim_batch * dim_set, embedding_dim)
        feat = self.image_encoder_block(embedding)
        feat = feat.reshape(dim_batch, dim_set, feat.shape[-1])
        feat = torch.cat([feat, onehot], dim=-1)

        feat = self.set_encoder(feat)
        feat = self.set_pooling(feat, set_size)
        feat = self.fspool_linear_layer(feat)

        feat = self.compatibility_layer(feat)
        return feat


class SlayNet(SlayNetImageOnly):
    def __init__(
            self,
            args: argparse.Namespace,
            number_of_types: int = 11,
            raw_img_embedding_dim: int = 512,
            raw_txt_embedding_dim: int = 512
    ):
        super(SlayNet, self).__init__(
            args,
            number_of_types=number_of_types,
            raw_img_embedding_dim=raw_img_embedding_dim
        )
        assert raw_txt_embedding_dim > 0
        assert self.text_embedding_size > 0
        self.text_encoder_block = nn.Sequential(
                nn.Linear(raw_txt_embedding_dim, 256, bias=False),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Linear(256, 128, bias=False),
                nn.ReLU(),
                nn.Linear(128, self.text_embedding_size, bias=False),
            )

    def forward(self, img_embedding, onehot, project_to_onehot, feat_mask=None, set_size=None, txt=None):
        dim_batch, dim_set, img_embedding_dim = img_embedding.shape
        _, _, txt_embedding_dim = txt.shape
        # merging batch and set size so that the tensor can be passed to backbone
        img_embedding = img_embedding.reshape(dim_batch * dim_set, img_embedding_dim)
        img_feat = self.image_encoder_block(img_embedding)
        img_feat = img_feat.reshape(dim_batch, dim_set, img_feat.shape[-1])

        txt = txt.reshape(dim_batch * dim_set, txt_embedding_dim)
        txt_feat = self.text_encoder_block(txt)
        txt_feat = txt_feat.reshape(dim_batch, dim_set, txt_feat.shape[-1])

        feat = torch.cat([img_feat, txt_feat, onehot], dim=-1)

        feat = self.set_layer(feat)
        feat = feat.squeeze(1)

        feat = self.csa_layer(feat, project_to_onehot)
        return feat

    def forward_positive(self, img_embedding, onehot, txt=None):
        img_feat = self.image_encoder_block(img_embedding)
        txt_feat = self.text_encoder_block(txt)
        feat = torch.cat([img_feat, txt_feat], dim=-1)
        feat = self.csa_layer(feat, onehot)
        return feat

    def forward_negative(self, img_embedding, onehot, txt=None):
        dim_batch, dim_set, img_embedding_dim = img_embedding.shape
        _, _, txt_embedding_dim = txt.shape
        # merging batch and set size so that the tensor can be passed to backbone
        img_embedding = img_embedding.reshape(dim_batch * dim_set, img_embedding_dim)
        img_feat = self.image_encoder_block(img_embedding)
        txt = txt.reshape(dim_batch * dim_set, txt_embedding_dim)
        txt_feat = self.text_encoder_block(txt)
        feat = torch.cat([img_feat, txt_feat], dim=-1)

        onehot = onehot.reshape(
            onehot.shape[0] * onehot.shape[1],
            onehot.shape[2]
        )
        feat = self.csa_layer(feat, onehot)
        feat = feat.reshape(dim_batch, dim_set, feat.shape[-1])
        return feat

    def forward_compatibility(self, img_embedding, onehot, feat_mask=None, set_size=None, txt=None):
        dim_batch, dim_set, img_embedding_dim = img_embedding.shape
        _, _, txt_embedding_dim = txt.shape
        # merging batch and set size so that the tensor can be passed to backbone
        img_embedding = img_embedding.reshape(dim_batch * dim_set, img_embedding_dim)
        img_feat = self.image_encoder_block(img_embedding)
        img_feat = img_feat.reshape(dim_batch, dim_set, img_feat.shape[-1])

        txt = txt.reshape(dim_batch * dim_set, txt_embedding_dim)
        txt_feat = self.text_encoder_block(txt)
        txt_feat = txt_feat.reshape(dim_batch, dim_set, txt_feat.shape[-1])

        feat = torch.cat([img_feat, txt_feat, onehot], dim=-1)

        feat = self.set_layer(feat)
        feat = feat.squeeze(1)

        feat = self.compatibility_layer(feat)
        return feat


class SlayNetFSPool(SlayNet):
    def __init__(
            self,
            args: argparse.Namespace,
            number_of_types: int = 11,
            raw_img_embedding_dim: int = 512,
            raw_txt_embedding_dim: int = 512
    ):
        assert args.set_pooling_type == 'FSPool'
        super(SlayNetFSPool, self).__init__(
            args,
            number_of_types=number_of_types,
            raw_img_embedding_dim=raw_img_embedding_dim,
            raw_txt_embedding_dim=raw_txt_embedding_dim
        )
        self.fspool_linear_layer = nn.Linear(args.set_hidden_dim, self.embedding_size)

    def forward(self, img_embedding, onehot, project_to_onehot, feat_mask=None, set_size=None, txt=None):
        dim_batch, dim_set, img_embedding_dim = img_embedding.shape
        _, _, txt_embedding_dim = txt.shape
        # merging batch and set size so that the tensor can be passed to backbone
        img_embedding = img_embedding.reshape(dim_batch * dim_set, img_embedding_dim)
        img_feat = self.image_encoder_block(img_embedding)
        img_feat = img_feat.reshape(dim_batch, dim_set, img_feat.shape[-1])

        txt = txt.reshape(dim_batch * dim_set, txt_embedding_dim)
        txt_feat = self.text_encoder_block(txt)
        txt_feat = txt_feat.reshape(dim_batch, dim_set, txt_feat.shape[-1])

        feat = torch.cat([img_feat, txt_feat, onehot], dim=-1)

        feat = self.set_encoder(feat)
        feat = self.set_pooling(feat, set_size)
        feat = self.fspool_linear_layer(feat)

        feat = self.csa_layer(feat, project_to_onehot)
        return feat

    def forward_compatibility(self, img_embedding, onehot, feat_mask=None, set_size=None, txt=None):
        dim_batch, dim_set, img_embedding_dim = img_embedding.shape
        _, _, txt_embedding_dim = txt.shape
        # merging batch and set size so that the tensor can be passed to backbone
        img_embedding = img_embedding.reshape(dim_batch * dim_set, img_embedding_dim)
        img_feat = self.image_encoder_block(img_embedding)
        img_feat = img_feat.reshape(dim_batch, dim_set, img_feat.shape[-1])

        txt = txt.reshape(dim_batch * dim_set, txt_embedding_dim)
        txt_feat = self.text_encoder_block(txt)
        txt_feat = txt_feat.reshape(dim_batch, dim_set, txt_feat.shape[-1])

        feat = torch.cat([img_feat, txt_feat, onehot], dim=-1)

        feat = self.set_encoder(feat)
        feat = self.set_pooling(feat, set_size)
        feat = self.fspool_linear_layer(feat)

        feat = self.compatibility_layer(feat)
        return feat




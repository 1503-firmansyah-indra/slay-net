# some parts from this script are taken from https://github.com/mvasil/fashion-compatibility/blob/master/polyvore_outfits.py

import argparse
from collections import defaultdict
import json
import os

import numpy as np
from PIL import Image

from loguru import logger
import torch
from torch.nn.utils.rnn import pad_sequence


class OutfitSetLoader(torch.utils.data.Dataset):
    def __init__(self, args, split, learning_type=None,
                 item_embeddings=None, item_embeddings_index_mapping=None,
                 item_text_embeddings=None):
        '''

        :param args:
            args.datadir: the root directory of the data
            args.polyvore_split: if the split of the data is "nondisjoint" or "disjoint"
        :param split:
            if the data is for "train", "valid" or "test"
        :param learning_type:
            to indicate if the dataset is going to be used for contrastive "contrastive" learning
            or compatibility "compatibility" learning
        '''

        self.image_embedding_size = args.dim_embed_img
        self.text_embedding_size = args.dim_embed_txt
        self.item_text_embeddings = item_text_embeddings
        if self.text_embedding_size > 0:
            logger.info(f"image and text data are used")
            assert self.item_text_embeddings is not None
        else:
            logger.info(f"only image data is used")

        self.item_embeddings = item_embeddings
        self.item_embeddings_index_mapping = item_embeddings_index_mapping

        self.neg_sample_size = args.negative_sample_size
        self.neg_sampling_strategy = 'general'

        self.max_set_len = args.max_set_len
        logger.info(f"Max set length: {self.max_set_len}")

        self.learning_type = learning_type

        self.rootdir = args.data_root_dir

        assert split in ["train", "valid", "test"]
        self.split = split
        self.is_train = split == 'train'

        if self.is_train:
            with open(args.contrastive_learning_data_path, 'r') as f:
                # TODO: the name of below instance attribute used to be "outfit_set_data", modify the rest of code accordingly
                self.contrastive_learning_data = json.load(f)
            '''
            the instance attribute "contrastive_learning_data" above has the following format
            list [
                <outfit_id>,
                [<positive image name>, <semantic category>],
                [
                    [<each_outfit image name>, <semantic category>], ...
                ]
            ]
            '''

        with open(args.item_types_path, 'r') as f:
            self.type_map = json.load(f)

        self.num_conditions = len(self.type_map.keys())

        # loading the raw outfits data based on the data split specified
        with open(os.path.join(self.rootdir, f"{split}.json")) as f:
            outfit_data = json.load(f)

        # loading the metadata of items
        with open(args.item_metadata_path, 'r') as f:
            meta_data = json.load(f)

        # get list of images and make a mapping used to quickly organize the data
        self.im2category = {}
        self.category2ims = {}
        self.full_outfit_set_data = {}

        self.im2finegrain = {}
        finegrain2im = defaultdict(set)

        # the variable "category" in the code block below refers to the general item type
        # the variable "finegrain" in the code block below refers to the fine-grained item type
        for outfit in outfit_data:
            outfit_id = outfit['set_id']
            self.full_outfit_set_data[outfit_id] = {}
            for item in outfit['items']:
                im = item['item_id']
                category = meta_data[im]['semantic_category']
                self.im2category[im] = category

                finegrain = meta_data[im]['category_id']
                self.im2finegrain[im] = finegrain
                finegrain2im[finegrain].add(im)

                if category not in self.category2ims:
                    self.category2ims[category] = {}

                if outfit_id not in self.category2ims[category]:
                    self.category2ims[category][outfit_id] = []

                self.category2ims[category][outfit_id].append(im)
                self.full_outfit_set_data[outfit_id][str(item['index'])] = im

        self.finegrain2im = {i: list(j) for i, j in finegrain2im.items()}
        self.finegrain2len = {i: len(j) for i, j in self.finegrain2im.items()}

        self.fitb_questions = None
        self.compatibility_questions = None
        if not self.is_train:
            self.fitb_questions = self.load_fitb_questions()
            self.compatibility_questions = self.load_compatibility_questions()

        # the variable below is to be set by class method
        #   "prepare_for_metrics_calculation"
        self.set_metric = None

    def prepare_for_metrics_calculation(self, input_metric):
        if self.is_train and self.fitb_questions is None:
            self.fitb_questions = self.load_fitb_questions()
        if self.is_train and self.compatibility_questions is None:
            self.compatibility_questions = self.load_compatibility_questions()
        assert input_metric in ['fitb', 'compatibility']
        self.set_metric = input_metric
        return True

    def prepare_for_training(self, learning_type):
        assert learning_type in ["contrastive", "compatibility"]
        self.set_metric = None
        self.learning_type = learning_type
        if self.learning_type == "compatibility" and self.compatibility_questions is None:
            self.compatibility_questions = self.load_compatibility_questions()
        return True

    def load_fitb_questions(self):
        with open(os.path.join(self.rootdir, f"fill_in_blank_{self.split}.json"), 'r') as f:
            raw_fitb_data = json.load(f)
        # TODO add note on data structure
        output = []
        # Note: index 0 under key 'answers' is the correct answer
        for each_fitb in raw_fitb_data:
            this_question = []
            for each_q in each_fitb['question']:
                outfit_id, item_idx = each_q.split('_')
                img = self.full_outfit_set_data[outfit_id][item_idx]
                item_type = self.im2category[img]
                this_question.append((img, item_type))
            this_answers = []
            for each_a in each_fitb['answers']:
                outfit_id, item_idx = each_a.split('_')
                img = self.full_outfit_set_data[outfit_id][item_idx]
                item_type = self.im2category[img]
                this_answers.append((img, item_type))
            output.append({
                'question': this_question,
                'answers': this_answers
            })
        return output

    def load_compatibility_questions(self):
        if self.split == 'train':
            if self.use_full_data:
                logger.info("loading FULL compatibility data")
                compatibility_data_path = os.path.join(self.rootdir, f"compatibility_{self.split}.txt")
            else:
                logger.info("loading PARTIAL compatibility data")
                compatibility_data_path = os.path.join(self.rootdir, f"compatibility_{self.split}_small.txt")
        elif self.split in ['valid', 'test']:
            compatibility_data_path = os.path.join(self.rootdir, f"compatibility_{self.split}.txt")
        else:
            raise Exception("Invalid argument 'split' ")
        with open(compatibility_data_path, 'r') as f:
            raw_comp_data = [i.strip().split() for i in f.readlines()]
        output = []
        for each_raw in raw_comp_data:
            this_items = []
            for each_item in each_raw[1:]:
                outfit_id, item_idx = each_item.split('_')
                img = self.full_outfit_set_data[outfit_id][item_idx]
                item_type = self.im2category[img]
                this_items.append((img, item_type))
            output.append((
                int(each_raw[0]),
                this_items
            ))
        return output

    def load_item(self, image_id, item_type):
        '''

        :param image_id:
        :param item_type:
        :return:
        '''
        img_idx = self.item_embeddings_index_mapping[image_id]
        img = torch.from_numpy(self.item_embeddings[img_idx, :])
        type_onehot = torch.zeros(self.num_conditions, dtype=torch.float32)
        type_onehot[self.type_map[item_type]] = 1
        return [img, type_onehot]

    def load_text(self, image_id):
        img_idx = self.item_embeddings_index_mapping[image_id]
        txt = torch.from_numpy(self.item_text_embeddings[img_idx, :])
        return txt

    def set_sampling_strategy(self, input_strategy):
        assert input_strategy in ['general', 'fine-grained']
        self.neg_sampling_strategy = input_strategy
        return True

    def sample_negative(self, outfit_id, item_id, item_type):
        if self.neg_sampling_strategy == 'general':
            self._sample_negative_general(outfit_id, item_id, item_type)
        elif self.neg_sampling_strategy == 'contrastive':
            self.sample_negative_finegrained(outfit_id, item_id, item_type)
        else:
            raise

    def sample_negative_general(self, outfit_id, item_id, item_type):
        """ Returns a randomly sampled item from a different set
            than the outfit at data_index, but of the same type as
            item_type

            data_index: index in self.data where the positive pair
                        of items was pulled from
            item_type: the coarse type of the item that the item
                       that was paired with the anchor
            sample_size: the number of negative sample per instance
        """
        candidate_sets = list(self.category2ims[item_type].keys())
        candidate_sets.remove(outfit_id)
        attempts = 0
        negative_samples = []
        while len(negative_samples) < self.neg_sample_size and attempts < 100:
            attempts += 1
            remaining_count = self.neg_sample_size - len(negative_samples)
            choices = np.random.choice(candidate_sets, size=remaining_count, replace=False)
            item_indexes = [
                np.random.choice(len(self.category2ims[item_type][each_choice])) for each_choice in choices]
            this_samples = [
                [self.category2ims[item_type][each_choice][item_index], item_type]
                for each_choice, item_index in zip(choices, item_indexes)
                if self.category2ims[item_type][each_choice][item_index] != item_id
            ]
            negative_samples += this_samples
        return negative_samples

    def sample_negative_finegrained(self, outfit_id, item_id, item_type):
        this_finegrain = self.im2finegrain[item_id]
        attempts = 0
        negative_samples = []
        this_finegrain_candidates = self.finegrain2im[this_finegrain]
        same_type_in_outfit = self.category2ims[item_type][outfit_id]
        if self.finegrain2len[this_finegrain] < self.neg_sample_size:
            for each_im in this_finegrain_candidates:
                if each_im not in same_type_in_outfit:
                    negative_samples.append([each_im, item_type])
            candidate_sets = list(self.category2ims[item_type].keys())
            candidate_sets.remove(outfit_id)
            while len(negative_samples) < self.neg_sample_size and attempts < 100:
                attempts += 1
                remaining_count = self.neg_sample_size - len(negative_samples)
                choices = np.random.choice(candidate_sets, size=remaining_count, replace=False)
                item_indexes = [
                    np.random.choice(len(self.category2ims[item_type][each_choice])) for each_choice in choices]
                this_samples = [
                    [self.category2ims[item_type][each_choice][item_index], item_type]
                    for each_choice, item_index in zip(choices, item_indexes)
                    if self.category2ims[item_type][each_choice][item_index] != item_id
                ]
                negative_samples += this_samples
        else:
            while len(negative_samples) < self.neg_sample_size and attempts < 100:
                attempts += 1
                remaining_count = self.neg_sample_size - len(negative_samples)
                choices = np.random.choice(this_finegrain_candidates, size=remaining_count, replace=False)
                this_samples = [
                    [each_choice, item_type]
                    for each_choice in choices
                    if each_choice not in same_type_in_outfit
                ]
                negative_samples += this_samples
        return negative_samples

    def __getitem__(self, index):
        '''
        :param index:
            the index of the dataset
        :return:
        '''

        if self.is_train:
            assert self.learning_type in ["contrastive", "compatibility"]

        if self.is_train and self.learning_type == "contrastive" and self.set_metric is None:
            self.getitem_contrastive(index)

        elif self.set_metric == 'fitb':
            self.getitem_fitb(index)

        elif self.set_metric == "compatibility" or \
                (self.is_train and self.learning_type == "compatibility" and self.set_metric is None):
            self.getitem_compatibility(index)

        else:
            raise Exception("invalid combination of instance variables")

    def getitem_contrastive(self, index):
        outfit_id, positive, rest_of_outfit = self.self.contrastive_learning_data[index]
        positive_image_id, positive_type = positive
        positive_image, positive_type_onehot = self.load_item(positive_image_id, positive_type)
        if self.text_embedding_size > 0:
            positive_text = self.load_text(positive_image_id)
        else:
            positive_text = torch.zeros(1)

        outfit_image = ()
        outfit_type_onehot = ()
        outfit_text = ()
        for each_feature in rest_of_outfit:
            this_image, this_onehot = self.load_item(*each_feature)
            outfit_image += (this_image,)
            outfit_type_onehot += (this_onehot,)
            if self.text_embedding_size > 0:
                this_text = self.load_text(each_feature[0])
            else:
                this_text = torch.zeros(1)
            outfit_text += (this_text,)
        if len(outfit_image) < self.max_set_len and self.is_train:
            outfit_img_mask = torch.cat((
                torch.ones(len(outfit_image), self.image_embedding_size),
                torch.zeros(self.max_set_len - len(outfit_image), self.image_embedding_size)
            ))
            for _ in range(self.max_set_len - len(outfit_image)):
                outfit_image += (torch.zeros(this_image.shape),)
                outfit_type_onehot += (torch.zeros(this_onehot.shape),)
                outfit_text += (torch.zeros(this_text.shape),)
        else:
            outfit_img_mask = torch.ones(len(outfit_image), self.image_embedding_size)
        outfit_image = torch.stack(outfit_image)
        outfit_type_onehot = torch.stack(outfit_type_onehot)
        outfit_text = torch.stack(outfit_text)
        outfit_len = torch.tensor(len(rest_of_outfit))

        negative_items = self.sample_negative(
            outfit_id, positive_image_id, positive_type
        )
        negative_image = ()
        negative_type_onehot = ()
        negative_text = ()
        for each_negative in negative_items:
            neg_image, neg_onehot = self.load_item(*each_negative)
            negative_image += (neg_image,)
            negative_type_onehot += (neg_onehot,)
            if self.text_embedding_size > 0:
                neg_text = self.load_text(each_negative[0])
            else:
                neg_text = torch.zeros(1)
            negative_text += (neg_text,)
        negative_image = torch.stack(negative_image)
        negative_type_onehot = torch.stack(negative_type_onehot)
        negative_text = torch.stack(negative_text)

        return positive_image, positive_type_onehot, \
            outfit_image, outfit_type_onehot, outfit_img_mask, \
            negative_image, negative_type_onehot, outfit_len, \
            positive_text, outfit_text, negative_text

    def getitem_fitb(self, index):
        this_fitb = self.fitb_questions[index]
        this_question = this_fitb['question']
        this_answers = this_fitb['answers']

        question_image = ()
        question_type_onehot = ()
        question_text = ()
        for each_q in this_question:
            each_q_image, each_q_onehot = self.load_item(*each_q)
            question_image += (each_q_image,)
            question_type_onehot += (each_q_onehot,)
            if self.text_embedding_size > 0:
                each_q_text = self.load_text(each_q[0])
            else:
                each_q_text = torch.zeros(0)
            question_text += (each_q_text,)
        question_image = torch.stack(question_image)
        question_type_onehot = torch.stack(question_type_onehot)
        question_text = torch.stack(question_text)
        question_img_mask = torch.ones(question_image.shape[0], self.image_embedding_size)
        outfit_len = torch.tensor(len(this_question))

        answers_image = ()
        answers_type_onehot = ()
        answers_text = ()
        for each_a in this_answers:
            each_a_image, each_a_onehot = self.load_item(*each_a)
            answers_image += (each_a_image,)
            answers_type_onehot += (each_a_onehot,)
            if self.text_embedding_size > 0:
                each_a_text = self.load_text(each_a[0])
            else:
                each_a_text = torch.zeros(1)
            answers_text += (each_a_text,)
        answers_image = torch.stack(answers_image)
        answers_type_onehot = torch.stack(answers_type_onehot)
        answers_text = torch.stack(answers_text)

        # variable "target_type_onehot" is created based on the first onehot in "this_answers" items list
        #  this is because the item types (and hence the onehot) of items in "this_answers" are the same
        target_type_onehot = answers_type_onehot[0, :]

        return question_image, question_type_onehot, question_img_mask, \
            target_type_onehot, answers_image, answers_type_onehot, outfit_len, \
            question_text, answers_text

    def getitem_compatibility(self, index):
        this_label, this_outfit = self.compatibility_questions[index]
        this_label = torch.tensor(this_label)

        outfit_image = ()
        outfit_type_onehot = ()
        outfit_text = ()
        for each_o in this_outfit:
            each_o_image, each_o_onehot = self.load_item(*each_o)
            outfit_image += (each_o_image,)
            outfit_type_onehot += (each_o_onehot,)
            if self.text_embedding_size > 0:
                each_o_text = self.load_text(each_o[0])
            else:
                each_o_text = torch.zeros(1)
            outfit_text += (each_o_text,)
        if len(outfit_image) < self.max_set_len and self.is_train:
            outfit_img_mask = torch.cat((
                torch.ones(len(outfit_image), self.image_embedding_size),
                torch.zeros(self.max_set_len - len(outfit_image), self.image_embedding_size)
            ))
            for _ in range(self.max_set_len - len(outfit_image)):
                outfit_image += (torch.zeros(each_o_image.shape),)
                outfit_type_onehot += (torch.zeros(each_o_onehot.shape),)
                outfit_text += (torch.zeros(each_o_text.shape),)
        else:
            outfit_img_mask = torch.ones(len(outfit_image), self.image_embedding_size)
        outfit_image = torch.stack(outfit_image)
        outfit_type_onehot = torch.stack(outfit_type_onehot)
        outfit_text = torch.stack(outfit_text)
        outfit_len = torch.tensor(len(this_outfit))
        return outfit_image, outfit_type_onehot, outfit_img_mask, this_label, outfit_len, outfit_text

    def __len__(self):
        if self.is_train and self.learning_type == "contrastive":
            return len(self.self.contrastive_learning_data)
        elif self.set_metric == "fitb":
            return len(self.fitb_questions)
        elif self.set_metric == "compatibility" or \
                (self.is_train and self.learning_type == "compatibility"):
            return len(self.compatibility_questions)
        else:
            raise Exception("invalid combination of instance variables")


class ItemEmbeddingLoader(OutfitSetLoader):
    def __init__(self, args: argparse.Namespace, split: str,
                 learning_type=None, item_embeddings=None,
                 item_embeddings_index_mapping=None, item_text_embeddings=None):
        assert split == 'test'

        self.rootdir = args.data_root_dir
        with open(os.path.join(self.rootdir, 'rak/finegrain2type.json'), 'r') as f:
            self.finegrain2type = json.load(f)

        self.selected_finegrain_list = list(self.finegrain2type.keys())

        super(ItemEmbeddingLoader, self).__init__(
            args, split, learning_type=learning_type, item_text_embeddings=item_text_embeddings,
            item_embeddings=item_embeddings, item_embeddings_index_mapping=item_embeddings_index_mapping
        )

        self.selected_finegrain = None
        self.selected_type = None
        self.selected_ims = None
        self.inference_mode = 'embedding'

    def prepare_for_inference(self, input_inference_mode: str):
        assert input_inference_mode in ['embedding', 'recall@k']
        self.inference_mode = input_inference_mode
        return True

    def set_finegrain(self, input_finegrain: str):
        assert self.inference_mode == 'embedding'
        assert input_finegrain in self.selected_finegrain_list
        self.selected_finegrain = input_finegrain
        self.selected_type = self.finegrain2type[input_finegrain]
        with open(os.path.join(self.rootdir, f"rak/{input_finegrain}"), 'r') as f:
            self.selected_ims = f.read().split()
        return True

    def load_fitb_questions(self):
        with open(os.path.join(self.rootdir, f"fill_in_blank_{self.split}.json"), 'r') as f:
            raw_fitb_data = json.load(f)
        output = []
        # Note: index 0 under key 'answers' is the correct answer
        skipped_count = 0
        included_count = 0
        for each_fitb in raw_fitb_data:
            a_outfit_id, a_item_idx = each_fitb['answers'][0].split('_')
            a_img = self.full_outfit_set_data[a_outfit_id][a_item_idx]
            a_finegrain = self.im2finegrain[a_img]
            a_item_type = self.im2type[a_img]

            if not a_finegrain in self.selected_finegrain_list:
                skipped_count += 1
                continue
            included_count += 1

            this_question = []
            for each_q in each_fitb['question']:
                outfit_id, item_idx = each_q.split('_')
                img = self.full_outfit_set_data[outfit_id][item_idx]
                item_type = self.im2type[img]
                this_question.append((img, item_type))

            output.append({
                'question': this_question,
                'answer_img_id': a_img,
                'answer_item_type': a_item_type,
                'answer_finegrain_type': a_finegrain
            })
        logger.info(f"FITB Eligible questions: {included_count}")
        logger.info(f"FITB Skipped questions: {skipped_count}")
        return output

    def __getitem__(self, index):
        if self.inference_mode == 'embedding':
            image_id = self.selected_ims[index]
            if self.text_embedding_size > 0:
                txt = self.load_text(image_id)
            else:
                txt = torch.zeros(1)
            image, type_onehot = self.load_item(image_id, self.selected_type)
            return image, txt, type_onehot, image_id, self.selected_finegrain
        elif self.inference_mode == 'recall@k':
            this_fitb = self.fitb_questions[index]
            this_question = this_fitb['question']

            question_image = ()
            question_type_onehot = ()
            question_text = ()
            for each_q in this_question:
                each_q_image, each_q_onehot = self.load_item(*each_q)
                question_image += (each_q_image,)
                question_type_onehot += (each_q_onehot,)
                if self.text_embedding_size > 0:
                    each_q_text = self.load_text(each_q[0])
                else:
                    each_q_text = torch.zeros(0)
                question_text += (each_q_text,)
            question_image = torch.stack(question_image)
            question_type_onehot = torch.stack(question_type_onehot)
            question_text = torch.stack(question_text)
            question_img_mask = torch.ones(question_image.shape[0], self.image_embedding_size)
            outfit_len = torch.tensor(len(this_question))

            answers_type_onehot = torch.zeros(self.num_conditions, dtype=torch.float32)
            answers_type_onehot[self.type_map[this_fitb['answer_item_type']]] = 1

            return question_image, question_type_onehot, question_img_mask, question_text, \
                outfit_len, answers_type_onehot, \
                this_fitb['answer_img_id'], this_fitb['answer_item_type'], this_fitb['answer_finegrain_type']
        else:
            raise Exception('invalid inference mode')

    def __len__(self):
        if self.inference_mode == 'embedding':
            return len(self.selected_ims)
        elif self.inference_mode == 'recall@k':
            return len(self.fitb_questions)
        else:
            raise Exception('invalid inference mode')


def triple_loss_set_collation(batch):
    pos_img = torch.stack([item[0] for item in batch])
    pos_onehot = torch.stack([item[1] for item in batch])
    outfit_img = pad_sequence([item[2] for item in batch], batch_first=True)
    outfit_onehot = pad_sequence([item[3] for item in batch], batch_first=True)
    outfit_img_mask = pad_sequence([item[4] for item in batch], batch_first=True)
    neg_img = torch.stack([item[5] for item in batch])
    neg_onehot = torch.stack([item[6] for item in batch])
    outfit_len = torch.stack([item[7] for item in batch])
    pos_txt = pad_sequence([item[8] for item in batch], batch_first=True)
    outfit_txt = pad_sequence([item[9] for item in batch], batch_first=True)
    neg_txt = pad_sequence([item[10] for item in batch], batch_first=True)
    return pos_img, pos_onehot, outfit_img, outfit_onehot, outfit_img_mask, \
        neg_img, neg_onehot, outfit_len, pos_txt, outfit_txt, neg_txt


def fitb_set_collation(batch):
    q_img = pad_sequence([item[0] for item in batch], batch_first=True)
    q_onehot = pad_sequence([item[1] for item in batch], batch_first=True)
    q_img_mask = pad_sequence([item[2] for item in batch], batch_first=True)
    target_onehot = torch.stack([item[3] for item in batch])
    a_img = torch.stack([item[4] for item in batch])
    a_onehot = torch.stack([item[5] for item in batch])
    outfit_len = torch.stack([item[6] for item in batch])
    q_txt = pad_sequence([item[7] for item in batch], batch_first=True)
    a_txt = torch.stack([item[8] for item in batch])
    return q_img, q_onehot, q_img_mask, target_onehot, a_img, a_onehot, outfit_len, q_txt, a_txt


def compatibility_set_collation(batch):
    outfit_img = pad_sequence([item[0] for item in batch], batch_first=True)
    outfit_onehot = pad_sequence([item[1] for item in batch], batch_first=True)
    outfit_img_mask = pad_sequence([item[2] for item in batch], batch_first=True)
    outfit_label = torch.stack([item[3] for item in batch])
    outfit_len = torch.stack([item[4] for item in batch])
    outfit_txt = pad_sequence([item[5] for item in batch], batch_first=True)
    return outfit_img, outfit_onehot, outfit_img_mask, outfit_label, outfit_len, outfit_txt


def item_embedding_collation(batch):
    img = torch.stack([item[0] for item in batch])
    txt = torch.stack([item[1] for item in batch])
    type_onehot = torch.stack([item[2] for item in batch])
    img_id = [item[3] for item in batch]
    finegrain = [item[4] for item in batch]
    return img, txt, type_onehot, img_id, finegrain


def recall_at_k_collation(batch):
    q_img = pad_sequence([item[0] for item in batch], batch_first=True)
    q_onehot = pad_sequence([item[1] for item in batch], batch_first=True)
    q_mask = pad_sequence([item[2] for item in batch], batch_first=True)
    q_text = pad_sequence([item[3] for item in batch], batch_first=True)
    q_outfit_len = torch.stack([item[4] for item in batch])
    a_onehot = torch.stack([item[5] for item in batch])
    a_img_id = [item[6] for item in batch]
    a_type = [item[7] for item in batch]
    a_finegrain = [item[8] for item in batch]
    return q_img, q_onehot, q_mask, q_text, q_outfit_len, a_onehot, a_img_id, a_type, a_finegrain



import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from vist import VIST


class VistDataset(data.Dataset):
    def __init__(self, image_dir, sis_path, dii_path, vocab, transform=None):
        self.image_dir = image_dir
        self.vist = VIST(sis_path, dii_path)
        self.ids = list(self.vist.stories.keys())
        self.vocab = vocab
        self.transform = transform


    def __getitem__(self, index):
        vist = self.vist
        vocab = self.vocab
        story_id = self.ids[index]

        targets = []
        images = []
        photo_sequence = []
        album_ids = []

        story = vist.stories[story_id]
        image_formats = ['.jpg', '.gif', '.png', '.bmp']
        for annotation in story:
            storylet_id = annotation["storylet_id"]
            image = Image.new('RGB', (256, 256))
            image_id = annotation["photo_flickr_id"]
            photo_sequence.append(image_id)
            album_ids.append(annotation["album_id"])
            for image_format in image_formats:
                try:
                    image = Image.open(os.path.join(self.image_dir, str(image_id) + image_format)).convert('RGB')
                except Exception:
                    continue

            if self.transform is not None:
                image = self.transform(image)

            images.append(image)

            text = annotation["text"]
            tokens = []
            try:
                tokens = nltk.tokenize.word_tokenize(text.lower())
            except Exception:
                pass

            caption = []
            caption.append(vocab('<start>'))
            caption.extend([vocab(token) for token in tokens])
            caption.append(vocab('<end>'))
            target = torch.Tensor(caption)
            targets.append(target)

        return torch.stack(images), targets, photo_sequence, album_ids


    def __len__(self):
        return len(self.ids)

    def GetItem(self, index): # this function is for visualizing
        vist = self.vist
        vocab = self.vocab
        story_id = self.ids[index]

        targets = [] # GT tensor
        images = []
        photo_sequence = []
        album_ids = []
        #added by sison
        descriptions = []
        phototags = []
        phototitles = [] 
        GT= []
        originalGT = []

        story = vist.stories[story_id]
        image_formats = ['.jpg', '.gif', '.png', '.bmp']
        
        for annotation in story:
            storylet_id = annotation["storylet_id"]
            image = Image.new('RGB', (256, 256))
            image_id = annotation["photo_flickr_id"]
            photo_sequence.append(image_id)
            album_ids.append(annotation["album_id"])
            for image_format in image_formats:
                try:
                    image = Image.open(os.path.join(self.image_dir, image_id + image_format)).convert('RGB')
                    break
                except Exception:
                    print('image exception occurred!', self.image_dir, image_id)
                    continue

            if self.transform is not None:
                image = self.transform(image)

            images.append(image)

            text = annotation["text"]
            og_text = annotation["original_text"]
            
            tokens = []
            try:
                tokens = nltk.tokenize.word_tokenize(text.lower())
            except Exception:
                pass

            caption = []
            caption.append(vocab('<start>'))
            caption.extend([vocab(token) for token in tokens])
            caption.append(vocab('<end>'))
            target = torch.Tensor(caption)
            targets.append(target)

            GT.append(text)
            originalGT.append(og_text)
            
        for photoid in photo_sequence:
            photoinfo_dict=vist.images[photoid]
            photodesc_dict=vist.descs

            phototags.append(photoinfo_dict['tags'])
            phototitles.append(photoinfo_dict['title'])
            descriptions.append(photodesc_dict[photoid])
            
        gift_ = descriptions, phototags, phototitles, GT, originalGT
        keys = 'descs', 'tags', 'titles', 'GTs', 'ogGTs'
        gift = dict(zip(keys,gift_))
        return images, targets, photo_sequence, album_ids, gift
               

    def GetLength(self):
        return len(self.ids)


def collate_fn(data):

    image_stories, caption_stories, photo_sequence_set, album_ids_set = zip(*data)

    targets_set = []
    lengths_set = []

    for captions in caption_stories:
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]

        targets_set.append(targets)
        lengths_set.append(lengths)

    return image_stories, targets_set, lengths_set, photo_sequence_set, album_ids_set


def get_loader(root, sis_path, vocab, transform, batch_size, shuffle, num_workers):
    vist = VistDataset(image_dir=root, sis_path=sis_path, vocab=vocab, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=vist, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return data_loader

import os
import numpy as np
import itertools
import collections
import torch
from .example import Example
from .utils import nostdout
from pycocotools.coco import COCO as pyCOCO
import re

class Dataset(object):
    def __init__(self, examples, fields):
        self.examples = examples
        self.fields = dict(fields)

    def collate_fn(self):
        def collate(batch):
            if len(self.fields) == 1:
                batch = [batch, ]
            else:
                batch = list(zip(*batch))

            tensors = []
            for field, data in zip(self.fields.values(), batch):
                tensor = field.process(data)
                if isinstance(tensor, collections.Sequence) and any(isinstance(t, torch.Tensor) for t in tensor):
                    tensors.extend(tensor)
                else:
                    tensors.append(tensor)

            if len(tensors) > 1:
                return tensors
            else:
                return tensors[0]

        return collate

    def __getitem__(self, i):
        example = self.examples[i]
        data = []
        for field_name, field in self.fields.items():
            data.append(field.preprocess(getattr(example, field_name)))

        if len(data) == 1:
            data = data[0]
        return data

    def __len__(self):
        return len(self.examples)

    def __getattr__(self, attr):
        if attr in self.fields:
            for x in self.examples:
                yield getattr(x, attr)


class ValueDataset(Dataset):
    def __init__(self, examples, fields, dictionary):
        self.dictionary = dictionary
        super(ValueDataset, self).__init__(examples, fields)

    def collate_fn(self):
        def collate(batch):
            value_batch_flattened = list(itertools.chain(*batch))
            value_tensors_flattened = super(ValueDataset, self).collate_fn()(value_batch_flattened)

            lengths = [0, ] + list(itertools.accumulate([len(x) for x in batch]))
            if isinstance(value_tensors_flattened, collections.Sequence) \
                    and any(isinstance(t, torch.Tensor) for t in value_tensors_flattened):
                value_tensors = [[vt[s:e] for (s, e) in zip(lengths[:-1], lengths[1:])] for vt in value_tensors_flattened]
            else:
                value_tensors = [value_tensors_flattened[s:e] for (s, e) in zip(lengths[:-1], lengths[1:])]

            return value_tensors
        return collate

    def __getitem__(self, i):
        if i not in self.dictionary:
            raise IndexError

        values_data = []
        for idx in self.dictionary[i]:
            value_data = super(ValueDataset, self).__getitem__(idx)
            values_data.append(value_data)
        return values_data

    def __len__(self):
        return len(self.dictionary)


class DictionaryDataset(Dataset):
    def __init__(self, examples, fields, key_fields):
        if not isinstance(key_fields, (tuple, list)):
            key_fields = (key_fields,)
        for field in key_fields:
            assert (field in fields)

        dictionary = collections.defaultdict(list)
        key_fields = {k: fields[k] for k in key_fields}
        value_fields = {k: fields[k] for k in fields.keys() if k not in key_fields}
        key_examples = []
        key_dict = dict()
        value_examples = []

        for i, e in enumerate(examples):
            key_example = Example.fromdict({k: getattr(e, k) for k in key_fields})
            value_example = Example.fromdict({v: getattr(e, v) for v in value_fields})
            if key_example not in key_dict:
                key_dict[key_example] = len(key_examples)
                key_examples.append(key_example)

            value_examples.append(value_example)
            dictionary[key_dict[key_example]].append(i)

        self.key_dataset = Dataset(key_examples, key_fields)
        self.value_dataset = ValueDataset(value_examples, value_fields, dictionary)
        super(DictionaryDataset, self).__init__(examples, fields)

    def collate_fn(self):
        def collate(batch):
            key_batch, value_batch = list(zip(*batch))
            key_tensors = self.key_dataset.collate_fn()(key_batch)
            value_tensors = self.value_dataset.collate_fn()(value_batch)
            return key_tensors, value_tensors
        return collate

    def __getitem__(self, i):
        return self.key_dataset[i], self.value_dataset[i]

    def __len__(self):
        return len(self.key_dataset)


def unique(sequence):
    seen = set()
    if isinstance(sequence[0], list):
        return [x for x in sequence if not (tuple(x) in seen or seen.add(tuple(x)))]
    else:
        return [x for x in sequence if not (x in seen or seen.add(x))]


class PairedDataset(Dataset):
    def __init__(self, examples, fields):
        assert ('image' in fields)
        assert ('text' in fields)
        super(PairedDataset, self).__init__(examples, fields)
        self.image_field = self.fields['image']
        self.text_field = self.fields['text']

    def image_set(self):
        img_list = [e.image for e in self.examples]
        image_set = unique(img_list)
        examples = [Example.fromdict({'image': i}) for i in image_set]
        dataset = Dataset(examples, {'image': self.image_field})
        return dataset

    def text_set(self):
        text_list = [e.text for e in self.examples]
        text_list = unique(text_list)
        examples = [Example.fromdict({'text': t}) for t in text_list]
        dataset = Dataset(examples, {'text': self.text_field})
        return dataset

    def image_dictionary(self, fields=None):
        if not fields:
            fields = self.fields
        dataset = DictionaryDataset(self.examples, fields, key_fields='image')
        return dataset

    def text_dictionary(self, fields=None):
        if not fields:
            fields = self.fields
        dataset = DictionaryDataset(self.examples, fields, key_fields='text')
        return dataset

    @property
    def splits(self):
        raise NotImplementedError


class COCO(PairedDataset):
    def __init__(self, image_field, text_field, img_root, ann_root, id_root=None, use_restval=True,
                 cut_validation=False, train_percentage=1, split_train_data=False):
        roots = {}
        roots['train'] = {
            'img': os.path.join(img_root, 'train2014'),
            'cap': os.path.join(ann_root, 'clean_train_coco.json')
        }
        # roots['val'] = {
        #     'img': os.path.join(img_root, 'val2014'),
        #     'cap': os.path.join(ann_root, 'captions_val2014.json')
        # }
        roots['val'] = {
            'img': os.path.join(img_root, 'train2014'),
            'cap': os.path.join(ann_root, 'clean_val_coco.json')
        }
        roots['test'] = {
            'img': os.path.join(img_root, 'val2014'),
            'cap': os.path.join(ann_root, 'captions_val2014.json')
        }
        roots['trainrestval'] = {
            'img': (roots['train']['img'], roots['val']['img']),
            'cap': (roots['train']['cap'], roots['val']['cap'])
        }

        if id_root is not None:
            ids = {}
            ids['train'] = np.load(os.path.join(id_root, 'coco_train_ids.npy'))


            ids['val'] = np.load(os.path.join(id_root, 'coco_dev_ids.npy'))
            if cut_validation:
                ids['val'] = ids['val'][:5000]
            ids['test'] = np.load(os.path.join(id_root, 'coco_test_ids.npy'))



            coco_restval_ids = np.load(os.path.join(id_root, 'coco_restval_ids.npy'))
            if split_train_data:
                np.random.shuffle(ids["train"])
                np.random.shuffle(coco_restval_ids)
                # ids["val"] = ids["train"][int(len(ids["train"])*train_percentage):]
                ids["train"] = ids["train"][:int(len(ids["train"])*train_percentage)]
                
                coco_restval_ids = coco_restval_ids[:int(len(coco_restval_ids)*train_percentage)]

            ids['trainrestval'] = (
                ids['train'],
                coco_restval_ids)

            if use_restval:
                roots['train'] = roots['trainrestval']
                ids['train'] = ids['trainrestval']


        else:
            ids = None

        with nostdout():
            self.train_examples, self.val_examples, self.test_examples = self.get_samples(roots, ids)
        examples = self.train_examples + self.val_examples + self.test_examples
        super(COCO, self).__init__(examples, {'image': image_field, 'text': text_field})

    @property
    def splits(self):
        train_split = PairedDataset(self.train_examples, self.fields)
        val_split = PairedDataset(self.val_examples, self.fields)
        test_split = PairedDataset(self.test_examples, self.fields)
        return train_split, val_split, test_split



    
    @classmethod
    def get_samples(cls, roots, ids_dataset=None):
        train_samples = []
        val_samples = []
        test_samples = []
        def is_not_english(sentence):
            """
            Returns False if the given sentence is in English, True otherwise.
            """
            # Regular expression pattern to match English letters and common punctuation marks
            pattern = r'^[A-Za-z]'
            # Check if the sentence matches the pattern
            return re.match(pattern, sentence) is None
        for split in ['train', 'val', 'test']:
            if isinstance(roots[split]['cap'], tuple):
                coco_dataset = (pyCOCO(roots[split]['cap'][0]), pyCOCO(roots[split]['cap'][1]))
                root = roots[split]['img']
            else:
                coco_dataset = (pyCOCO(roots[split]['cap']),)
                root = (roots[split]['img'],)

            if ids_dataset is None:
                ids = list(coco_dataset.anns.keys())
            else:
                ids = ids_dataset[split]

            if isinstance(ids, tuple):
                bp = len(ids[0])
                ids = list(ids[0]) + list(ids[1])
            else:
                bp = len(ids)

            for index in range(len(ids)):
                if index < bp:
                    coco = coco_dataset[0]
                    img_root = root[0]
                else:
                    coco = coco_dataset[1]
                    img_root = root[1]

                ann_id = ids[index]
                try:
                    caption = coco.anns[ann_id]['caption']
                    img_id = coco.anns[ann_id]['image_id']
                    filename = coco.loadImgs(img_id)[0]['file_name']
                except:
                    print("a missing annotation"+str(ann_id))
                    continue

                example = Example.fromdict({'image': os.path.join(img_root, filename), 'text': caption})
                #fix the function by adding for loop
                if is_not_english(example.text) and is_not_english(example.text[1::]) and is_not_english(example.text[2::]) and is_not_english(example.text[3::]):
                    if split == 'train':
                        train_samples.append(example)
                    elif split == 'val':
                        val_samples.append(example)
                    elif split == 'test':
                        test_samples.append(example)

        return train_samples, val_samples, test_samples

# import pandas as pd
import h5py
from PIL import Image
# import io
import json
class XM3600(PairedDataset):
    def __init__(self, image_field, text_field, img_root, ann_root):
        
        self.cross_modal = pd.read_csv(ann_root)
        self.root_dir = img_root
        examples = []
        caption = self.cross_modal.iloc[:, 1]
        caption = list(caption)
        image_id = self.cross_modal.iloc[:, 0]
        image_id = list(image_id)
        for id, cap in zip(image_id, caption):
           
           img_name = os.path.join(self.root_dir,
                                str(id)+".jpg")

           image = Image.open(img_name)
           image = np.array(image)
           example = Example.fromdict({'image': image, 'text': cap})
           examples.append(example)
        super(XM3600, self).__init__(examples, {'image': image_field, 'text': text_field})


import time

    
class CC3M(PairedDataset):
    def __init__(self, image_field, text_field, img_root, ann_root):

#Try to read the entire dataset into examples and calculate the run time
#Then try to use this code to load data into dictionary
        # start = time.time()
        # data_dict = {}
        # with h5py.File(img_root, "r") as file:
        #     datasets = file["/"]
        #     for dataset_name, dataset in file.items():
        #         dataset = np.asarray(dataset)
        #         img = Image.open(io.BytesIO(dataset))
        #         data_dict[dataset_name] = img[()]

        # end = time.time()
        # runtime = end-start
        # hf = h5py.File(img_root, 'r') # open a hdf5 file
        
        with open(ann_root) as f:
    # Load the JSON data from the file
             data = json.load(f)
        data = data["annotations"]
   

        examples = []

        for sample in  data:
           id = sample["image_id"]
           cap = sample["caption"] 
  
           example = Example.fromdict({'image': id, 'text': cap})
           examples.append(example)
           if len(cap)<4:
               test = 1




        super(CC3M, self).__init__(examples, {'image': image_field, 'text': text_field})




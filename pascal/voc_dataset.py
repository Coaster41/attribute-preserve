import torch
from torchvision import datasets, transforms
import numpy as np
import torch
from torchvision import datasets
from xml.etree.ElementTree import Element as ET_Element
from typing import Any, Dict
import collections

device = torch.device(f"cuda:0")
class VOCnew(datasets.VOCDetection):
    classes = ('aeroplane', 'bicycle', 'bird', 'boat',
                    'bottle', 'bus', 'car', 'cat', 'chair',
                    'cow', 'diningtable', 'dog', 'horse',
                    'motorbike', 'person', 'pottedplant',
                    'sheep', 'sofa', 'train', 'tvmonitor')
    class_to_ind = dict(zip(classes, range(len(classes))))   

    @staticmethod
    def parse_voc_xml(node: ET_Element) -> Dict[str, Any]:
        

        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(datasets.VOCDetection.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
                objs = [def_dic["object"]]
                lbl = np.zeros(len(VOCnew.classes))
                for ix, obj in enumerate(objs[0][0]):        
                    obj_class = VOCnew.class_to_ind[obj['name']]
                    lbl[obj_class] = 1
                return lbl
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

class Data:
    def __init__(self):
        train_dataset = VOCnew(root=r'/tmp/public_dataset/pytorch/pascalVOC-data', image_set='train', download=True,
                        transform=transforms.Compose([
                            transforms.Resize(330),
                            transforms.Pad(30),
                            transforms.RandomCrop(300),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ]))

        test_dataset = VOCnew(root=r'/tmp/public_dataset/pytorch/pascalVOC-data', image_set='val', download=False,
                        transform=transforms.Compose([
                            transforms.Resize(330), 
                            transforms.CenterCrop(300),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ]))
        self.loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
        self.loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

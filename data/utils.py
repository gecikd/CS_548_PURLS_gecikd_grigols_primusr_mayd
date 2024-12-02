import torch
import torch.nn.functional as F
import numpy as np
import os

splits = {'ntu': [  
                [2,3,4,8,20], # head interpolate + 7
                [4,5,6,7,8,9,10,11,21,22,23,24], #hands
                [0,1,4,8,12,16,20], # torso  interpolate + 5
                [0,12,13,14,15,16,17,18,19] # feet  interpolate + 3
            ],
          'kinetic':[
              [16,14,15,17,0],
              [1,2,3,4,5,6,7],
              [0,1,2,5,8,11],
              [8,11,9,12,10,13]
          ]}

def joint_interpolate(x):
    x = torch.Tensor(x)
    x2 = F.interpolate(x, (72, 12), mode='bilinear')
    x2 =x2.numpy()
    return x2

import os
import numpy as np

def load_labels(root, split, dataloader, model_name):
    if dataloader is None:
        raise ValueError("The 'dataloader' parameter is None. Please provide a valid dataloader string.")
    
    try:
        cls_num = int(dataloader.split('_')[-1])
    except (ValueError, IndexError):
        raise ValueError("The 'dataloader' parameter does not contain a valid class number.")

    emb_dim = 512
    
    unseen_inds_path = os.path.join(root, f'resources/label_splits/ru{split}.npy')
    seen_inds_path = os.path.join(root, f'resources/label_splits/rs{cls_num - split}.npy')

    try:
        unseen_inds = np.load(unseen_inds_path)
        seen_inds = np.load(seen_inds_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required file not found: {e.filename}")
    
    cls_labels_path = os.path.join(root, f'resources/ntu{cls_num}_bpnames.npy')
    
    try:
        cls_labels = np.load(cls_labels_path, allow_pickle=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Class labels file not found: {cls_labels_path}")
    
    return cls_num, emb_dim, unseen_inds, seen_inds, cls_labels

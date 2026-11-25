import pickle
import numpy as np
from PIL import Image, ImageDraw
import os
import json


def within_margin(v):
    if v[0]>16 and v[0]<256-16 and v[1]>16 and v[1]<256-16:
        return True
    return False

def convert_pred_RNGDet(dir):
    with open('../data/data_split.json','r') as jf:
        tile_list = json.load(jf)['test']
    
    for tile_idx in tile_list:
        print(tile_idx)
        
        gt_graph = f'../{dir}/test/graph/{tile_idx}.p'
        gt_graph = pickle.load(open(gt_graph, "rb"), encoding='latin1')
        new_graph = {}
        for n, v in gt_graph.items():
            if within_margin(n):
                new_graph[(n[0]-16,n[1]-16)] = [(u[0]-16,u[1]-16) for u in v if within_margin(u)]
        
        pickle.dump(new_graph,open(f'../{dir}/test/graph/{tile_idx}_crop.p','wb'),protocol=2)
    
try:
    convert_pred_RNGDet('RNGDet_multi_ins')
except:
    pass

try:
    convert_pred_RNGDet('RNGDet')
except:
    pass

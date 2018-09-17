import numpy as np


palette = [[128,64,128], [244,35,232], [70,70,70],   [102,102,156], [190,153,153],
            [153,153,153],[250,170,30], [220,220,0],  [107,142,35], [152,251,152],
            [70,130,180], [220,20,60],  [255,0,0],    [0,0,142],    [0,0,70],
            [0,60,100],   [0,80,100],   [0,0,230],    [119,11,32],
            [0,0,0], [0,0,0]]
palette_cid_int8 = np.copy(palette)
palette = np.array(palette)/255.

class_names = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle", "void"]


def make_palette():
    for lid in [-1, -1, -1, -1, -1, -1, -1, 0, 1, -1, -1, 2, 3, 4, -1, -1, -1, 5,
                -1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1, -1, 16, 17, 18, -1]:
        lid_ = lid if lid is not -1 else 19
        yield palette[lid_]


palette_extended = np.array(list(make_palette()))
palette_lid = palette_extended
palette_cid = palette


def im2lbl_path(path, dataset):

    if dataset == 'gta5':
        return path.replace('images', 'labels')
    elif dataset == 'cityscapes':
        return path.replace('leftImg8bit.png', 'gtFine_labelIds.png').replace('leftImg8bit/', 'gtFine/')
    elif dataset == 'mapillary':
        return path.replace('images', 'labels').replace('.jpg', '.png')

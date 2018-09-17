from collections import namedtuple

# #--------------------------------------------------------------------------------
# # Definitions
# #--------------------------------------------------------------------------------
#
# # a label and all meta information
# Label = namedtuple('Label', [
#     'name', 'clsId', 'id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval', 'color'])
#
# ## CHANGE CLASS ID BELOW
#
# labels = [
#     #     name                    hexId    lid   clsId   category  catId  hasInstance ignoreInEval   color
#     Label('others'              ,    0 ,    0,   -1   ,  '其他'    ,   0  ,False , True  , 0x000000 ),
#     Label('rover'               , 0x01 ,    1,    0   ,  '其他'    ,   0  ,False , True  , 0X000000 ),
#     Label('sky'                 , 0x11 ,   17,    1    , '天空'    ,   1  ,False , False , 0x4682B4 ),
#     Label('car'                 , 0x21 ,   33,    2    , '移动物体',   2  ,True  , False , 0x00008E ),
#     Label('car_groups'          , 0xA1 ,  161,    2    , '移动物体',   2  ,True  , False , 0x00008E ),
#     Label('motorbicycle'        , 0x22 ,   34,    3    , '移动物体',   2  ,True  , False , 0x0000E6 ),
#     Label('motorbicycle_group'  , 0xA2 ,  162,    3    , '移动物体',   2  ,True  , False , 0x0000E6 ),
#     Label('bicycle'             , 0x23 ,   35,    4    , '移动物体',   2  ,True  , False , 0x770B20 ),
#     Label('bicycle_group'       , 0xA3 ,  163,    4    , '移动物体',   2  ,True  , False , 0x770B20 ),
#     Label('person'              , 0x24 ,   36,    5    , '移动物体',   2  ,True  , False , 0x0080c0 ),
#     Label('person_group'        , 0xA4 ,  164,    5    , '移动物体',   2  ,True  , False , 0x0080c0 ),
#     Label('rider'               , 0x25 ,   37,    6    , '移动物体',   2  ,True  , False , 0x804080 ),
#     Label('rider_group'         , 0xA5 ,  165,    6    , '移动物体',   2  ,True  , False , 0x804080 ),
#     Label('truck'               , 0x26 ,   38,    7    , '移动物体',   2  ,True  , False , 0x8000c0 ),
#     Label('truck_group'         , 0xA6 ,  166,    7    , '移动物体',   2  ,True  , False , 0x8000c0 ),
#     Label('bus'                 , 0x27 ,   39,    8    , '移动物体',   2  ,True  , False , 0xc00040 ),
#     Label('bus_group'           , 0xA7 ,  167,    8    , '移动物体',   2  ,True  , False , 0xc00040 ),
#     Label('tricycle'            , 0x28 ,   40,    9    , '移动物体',   2  ,True  , False , 0x8080c0 ),
#     Label('tricycle_group'      , 0xA8 ,  168,    9    , '移动物体',   2  ,True  , False , 0x8080c0 ),
#     Label('road'                , 0x31 ,   49,    10   , '平面'   ,   3  ,False , False , 0xc080c0 ),
#     Label('sidewalk'            , 0x32 ,   50,    11   , '平面'   ,   3  ,False , False , 0xc08040 ),
#     Label('traffic_cone'        , 0x41 ,   65,    12   , '路间障碍',   4  ,False , False , 0x000040 ),
#     Label('road_pile'           , 0x42 ,   66,    13   , '路间障碍',   4  ,False , False , 0x0000c0 ),
#     Label('fence'               , 0x43 ,   67,    14   , '路间障碍',   4  ,False , False , 0x404080 ),
#     Label('traffic_light'       , 0x51 ,   81,    15   , '路边物体',   5  ,False , False , 0xc04080 ),
#     Label('pole'                , 0x52 ,   82,    16   , '路边物体',   5  ,False , False , 0xc08080 ),
#     Label('traffic_sign'        , 0x53 ,   83,    17   , '路边物体',   5  ,False , False , 0x004040 ),
#     Label('wall'                , 0x54 ,   84,    18   , '路边物体',   5  ,False , False , 0xc0c080 ),
#     Label('dustbin'             , 0x55 ,   85,    19   , '路边物体',   5  ,False , False , 0x4000c0 ),
#     Label('billboard'           , 0x56 ,   86,    20   , '路边物体',   5  ,False , False , 0xc000c0 ),
#     Label('building'            , 0x61 ,   97,    21   , '建筑'   ,   6  ,False , False , 0xc00080 ),
#     Label('bridge'              , 0x62 ,   98,    22   , '建筑'    ,   6  ,False , True  , 0x808000 ),
#     Label('tunnel'              , 0x63 ,   99,    23   , '建筑'    ,   6  ,False , True  , 0x800000 ),
#     Label('overpass'            , 0x64 ,  100,    24   , '建筑'    ,   6  ,False , True  , 0x408040 ),
#     Label('vegetation'          , 0x71 ,  113,    25   , '自然'    ,   7  ,False , False , 0x808040 ),
#     Label('unlabeled'           , 0xFF ,  255,    -1   , '未标注'  ,   8  ,False , True  , 0xFFFFFF ),
# ]

# labels = [
#     #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
#     Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'dynamic'              ,  1 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
#     Label(  'ego vehicle'          ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'ground'               ,  3 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
#     Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'parking'              ,  5 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
#     Label(  'rail track'           ,  6 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
#     Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
#     Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
#     Label(  'bridge'               ,  9 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
#     Label(  'building'             , 10 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
#     Label(  'fence'                , 11 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
#     Label(  'garage'               , 12 ,      255 , 'construction'    , 2       , False        , True         , (180,100,180) ),
#     Label(  'guard rail'           , 13 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
#     Label(  'tunnel'               , 14 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
#     Label(  'wall'                 , 15 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
#     Label(  'banner'               , 16 ,      255 , 'object'          , 3       , False        , True         , (250,170,100) ),
#     Label(  'billboard'            , 17 ,      255 , 'object'          , 3       , False        , True         , (220,220,250) ),
#     Label(  'lane divider'         , 18 ,      255 , 'object'          , 3       , False        , True         , (255, 165, 0) ),
#     Label(  'parking sign'         , 19 ,      255 , 'object'          , 3       , False        , False        , (220, 20, 60) ),
#     Label(  'pole'                 , 20 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
#     Label(  'polegroup'            , 21 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
#     Label(  'street light'         , 22 ,      255 , 'object'          , 3       , False        , True         , (220,220,100) ),
#     Label(  'traffic cone'         , 23 ,      255 , 'object'          , 3       , False        , True         , (255, 70,  0) ),
#     Label(  'traffic device'       , 24 ,      255 , 'object'          , 3       , False        , True         , (220,220,220) ),
#     Label(  'traffic light'        , 25 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
#     Label(  'traffic sign'         , 26 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
#     Label(  'traffic sign frame'   , 27 ,      255 , 'object'          , 3       , False        , True         , (250,170,250) ),
#     Label(  'terrain'              , 28 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
#     Label(  'vegetation'           , 29 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
#     Label(  'sky'                  , 30 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
#     Label(  'person'               , 31 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
#     Label(  'rider'                , 32 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
#     Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
#     Label(  'bus'                  , 34 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
#     Label(  'car'                  , 35 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
#     Label(  'caravan'              , 36 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
#     Label(  'motorcycle'           , 37 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
#     Label(  'trailer'              , 38 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
#     Label(  'train'                , 39 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
#     Label(  'truck'                , 40 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
# ]

from collections import namedtuple

# a label and all meta information
# Code inspired by Cityscapes https://github.com/mcordts/cityscapesScripts
Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class

    'id',  # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # evaluation server.

    'trainId',
    # Feel free to modify these IDs as suitable for your method. Then create
    # ground truth images with train IDs, using the tools provided in the
    # 'preparation' folder. However, make sure to validate or submit results
    # to our evaluation server using the regular IDs above!
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!

    'category',  # The name of the category that this label belongs to

    'categoryId',
    # The ID of this category. Used to create ground truth images
    # on category level.

    'hasInstances',
    # Whether this label distinguishes between single instances or not

    'ignoreInEval',
    # Whether pixels having this class as ground truth label are ignored
    # during evaluations or not

    'color',  # The color of this label
])


# Our extended list of label types. Our train id is compatible with Cityscapes
labels = [
    #       name                     id    cid       category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            , 255 ,       19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              , 255 ,       19 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ego vehicle'          , 255 ,       19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ground'               , 255 ,       19 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'static'               , 255 ,       19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'parking'              , 255 ,       19 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 255 ,       19 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'road'                 ,   0 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,   1 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'bridge'               , 255 ,       19 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'building'             ,   2 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'fence'                ,   4 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'garage'               , 255 ,       19 , 'construction'    , 2       , False        , True         , (180,100,180) ),
    Label(  'guard rail'           , 255 ,       19 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'tunnel'               , 255 ,       19 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'wall'                 ,   3 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'banner'               , 255 ,       19 , 'object'          , 3       , False        , True         , (250,170,100) ),
    Label(  'billboard'            , 255 ,       19 , 'object'          , 3       , False        , True         , (220,220,250) ),
    Label(  'lane divider'         , 255 ,       19 , 'object'          , 3       , False        , True         , (255, 165, 0) ),
    Label(  'parking sign'         , 255 ,       19 , 'object'          , 3       , False        , False        , (220, 20, 60) ),
    Label(  'pole'                 ,   5 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 255 ,       19 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'street light'         , 255 ,       19 , 'object'          , 3       , False        , True         , (220,220,100) ),
    Label(  'traffic cone'         , 255 ,       19 , 'object'          , 3       , False        , True         , (255, 70,  0) ),
    Label(  'traffic device'       , 255 ,       19 , 'object'          , 3       , False        , True         , (220,220,220) ),
    Label(  'traffic light'        ,   6 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         ,   7 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'traffic sign frame'   , 255 ,       19 , 'object'          , 3       , False        , True         , (250,170,250) ),
    Label(  'terrain'              ,   9 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'vegetation'           ,   8 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'sky'                  ,  10 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               ,  11 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                ,  12 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'bicycle'              ,  18 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'bus'                  ,  15 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'car'                  ,  13 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'caravan'              , 255 ,       19 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'motorcycle'           ,  17 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'trailer'              , 255 ,       19 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                ,  16 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'truck'                ,  14 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
]



## DO NOT CHANGE BELOW


ids_total = list()
names_total = list()
colors_total = list()
class_ids_total = list()
for label in labels:
    ids_total.append(label.id)
    names_total.append(label.name)
    colors_total.append(label.color)
    class_ids_total.append(label.trainId)

lids2cids = list()

for i in range(max(ids_total)+1):
    if i in ids_total:
        index = ids_total.index(i)
        lids2cids.append(class_ids_total[index])
    else:
        lids2cids.append(-1)

print('lids2cids:')
print(str(lids2cids) + '\n')

cids2colors = list()
cids2labels = list()
cids2lids = list()

for x in range(max(lids2cids)+1):
    try:
        index = class_ids_total.index(x)
        color_tmp = colors_total[index]
        cids2colors.append(color_tmp)
        cids2labels.append(names_total[index])
        cids2lids.append(ids_total[index])
    except ValueError:
        print('Not all class labels from 0 up to ' + str(max(lids2cids)) + ' exist.')
        print('cids2colors, cids2labels and cids2lids could not be computed! \n')

print('cids2colors:')
print(str(cids2colors) + '\n')

print('cids2labels:')
print(str(cids2labels) + '\n')

print('cids2lids:')
print(str(cids2lids) + '\n')

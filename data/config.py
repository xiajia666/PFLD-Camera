# config.py

cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]], # 三个分别对应三个有效的特征层，比较浅的特征层先验框的基础边长小。深的特征层更适合检测大物体
    'steps': [8, 16, 32, 64], # 比较浅的有效特征层对输入进来的图片长和宽压缩三次，变为原来的1/8. 深特征层进行5次压缩，长和宽变为原来的1/32
    'variance': [0.1, 0.2],
    'clip': True,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 1,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': False,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3, },
    'in_channel': 32,
    'out_channel': 64
}

cfg_re50 = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': True,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 1,
    'ngpu': 1,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': False,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}
cfg_gnet = {
    'name': 'ghostnet',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': True,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 16,
    'ngpu': 1,
    'epoch': 100,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': False,
    'return_layers': {'blocks1': 1, 'blocks2': 2, 'blocks3': 3},
    # 'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}
# cfg_mnetv3 = {
#     'name': 'mobilev3',
#     'min_sizes': [[16, 25.40], [32, 50.80], [64, 128], [256, 512]],
#     'steps': [4, 8, 16, 32],
#     'variance': [0.1, 0.2],
#     'clip': True,
#     'loc_weight': 2,
#     'gpu_train': True,
#     'batch_size': 16,
#     'ngpu': 1,
#     'epoch': 350,
#     'decay1': 190,
#     'decay2': 220,
#     'image_size': 640,
#     'pretrain': False,
#     'return_layers': {'bneck1': 1, 'bneck2': 2, 'bneck3': 3, 'bneck4': 4},
#     'in_channel': 32,
#     'out_channel': 64
# }

cfg_mnetv3 = {
    'name': 'mobilev3',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': True,
    'loc_weight': 2,
    'gpu_train': True,
    'batch_size': 16,
    'ngpu': 1,
    'epoch': 350,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': False,
    'return_layers': {'bneck1': 1, 'bneck2': 2, 'bneck3': 3},
    'in_channel': 32,
    'out_channel': 64
}

cfg_convlarge = {
    'name': 'convnexttiny',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': True,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 2,
    'ngpu': 1,
    'epoch': 1,
    'decay1': 190,
    'decay2': 220,
    'image_size': 680,
    'pretrain': False,
    'return_layers': {'stages1': 1, 'stages2': 2, 'stages3': 3},
    'in_channel': 96,
    'out_channel': 192
}
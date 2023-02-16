dataset_info = dict(
    dataset_name='pig',
    paper_info=dict(
    ),
    keypoint_info={
        0:
        dict(name='center', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(
            name='lelbow',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap='relbow'),
        2:
        dict(
            name='lwrist',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap='rwrist'),
        3:
        dict(
            name='lhand',
            id=3,
            color=[51, 153, 255],
            type='upper',
            swap='rhand'),
        4:
        dict(
            name='lfhoof',
            id=4,
            color=[51, 153, 255],
            type='upper',
            swap='rfhoof'),
        5:
        dict(
            name='relbow',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='lelbow'),
        6:
        dict(
            name='rwrist',
            id=6,
            color=[255, 128, 0],
            type='upper',
            swap='lwrist'),
        7:
        dict(
            name='rhand',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap='lhand'),
        8:
        dict(
            name='rfhoof',
            id=8,
            color=[255, 128, 0],
            type='upper',
            swap='lfhoof'),
        9:
        dict(
            name='lknee',
            id=9,
            color=[0, 255, 0],
            type='upper',
            swap='rknee'),
        10:
        dict(
            name='lankle',
            id=10,
            color=[255, 128, 0],
            type='upper',
            swap='rankle'),
        11:
        dict(
            name='lmetatarsus',
            id=11,
            color=[0, 255, 0],
            type='lower',
            swap='rmetatarsus'),
        12:
        dict(
            name='lrhoof',
            id=12,
            color=[255, 128, 0],
            type='lower',
            swap='rrhoof'),
        13:
        dict(
            name='rknee',
            id=13,
            color=[0, 255, 0],
            type='lower',
            swap='lknee'),
        14:
        dict(
            name='rankle',
            id=14,
            color=[255, 128, 0],
            type='lower',
            swap='lankle'),
        15:
        dict(
            name='rmetatarsus',
            id=15,
            color=[0, 255, 0],
            type='lower',
            swap='lmetatarsus'),
        16:
        dict(
            name='rrhoof',
            id=16,
            color=[255, 128, 0],
            type='lower',
            swap='lrhoof')
    },
    skeleton_info={
        0:
        dict(link=('center', 'lelbow'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('lelbow', 'lwrist'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('lwrist', 'lhand'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('lhand', 'lfhoof'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('center', 'relbow'), id=4, color=[51, 153, 255]),
        5:
        dict(link=('relbow', 'rwrist'), id=5, color=[51, 153, 255]),
        6:
        dict(link=('rwrist', 'rhand'), id=6, color=[51, 153, 255]),
        7:
        dict(
            link=('rhand', 'rfhoof'),
            id=7,
            color=[51, 153, 255]),
        8:
        dict(link=('center', 'lknee'), id=8, color=[0, 255, 0]),
        9:
        dict(
            link=('lknee', 'lankle'), id=9, color=[255, 128, 0]),
        10:
        dict(link=('lankle', 'lmetatarsus'), id=10, color=[0, 255, 0]),
        11:
        dict(link=('lmetatarsus', 'lrhoof'), id=11, color=[255, 128, 0]),
        12:
        dict(link=('center', 'rknee'), id=12, color=[51, 153, 255]),
        13:
        dict(link=('rknee', 'rankle'), id=13, color=[51, 153, 255]),
        14:
        dict(link=('rankle', 'rmetatarsus'), id=14, color=[51, 153, 255]),
        15:
        dict(link=('rmetatarsus', 'rrhoof'), id=15, color=[51, 153, 255]),

    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5,
        1.5
    ],
    sigmas=[
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
        0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
    ])

import json
import os.path as osp
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
from mmengine.utils import check_file_exist
from mmengine.config import Config
from mmpose.registry import DATASETS
from mmpose.datasets.datasets.base import BaseCocoStyleDataset


@DATASETS.register_module()
class PigPoseDataset(BaseCocoStyleDataset):
        METAINFO: dict = dataset_info


def get_cfg(args):
    # 模型 config 配置文件
    cfg = Config.fromfile(args.config)


    # # 基础配置
    cfg.data_root = args.data_root
    cfg.work_dir = args.work_dir
    cfg.gpu_ids = range(1)
    cfg.seed = 42
    cfg.model.test_cfg.output_heatmaps=True #输出热图
    cfg.dataset_type='PigPoseDataset'

    # 评估指标
    cfg.default_hooks.checkpoint.interval = 20
    cfg.default_hooks.checkpoint.save_best='coco/AP'


    cfg.model.head.out_channels=17 #17类
    # 学习率和训练策略
    cfg.param_scheduler = [
        dict(
            type='LinearLR', begin=0, end=500, start_factor=0.001,
            by_epoch=False),  # warm-up
        dict(
            type='MultiStepLR',
            begin=0,
            end=210,
            milestones=[170, 200],
            gamma=0.1,
            by_epoch=True)
    ]
    cfg.optim_wrapper = dict(optimizer=dict(type='Adam', lr=0.0005))


    cfg.train_cfg = dict(by_epoch=True, max_epochs=210, val_interval=2)

    # # 数据集配置
    cfg.train_dataloader = dict(
        batch_size=6,
        num_workers=2,
        persistent_workers=True,
        sampler=dict(type='DefaultSampler', shuffle=True),
        dataset=dict(
            type=cfg.dataset_type,
            data_root=cfg.data_root,
            data_mode=cfg.data_mode,
            ann_file='train.json',
            data_prefix=dict(img='train'),
            pipeline=cfg.train_pipeline,
        ))
    cfg.val_dataloader = dict(
        batch_size=6,
        num_workers=2,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
        dataset=dict(
            type=cfg.dataset_type,
            data_root=cfg.data_root,
            data_mode=cfg.data_mode,
            ann_file='val.json',
            data_prefix=dict(img='val'),
            test_mode=True,
            pipeline=cfg.val_pipeline,
        ))
    cfg.test_dataloader = cfg.val_dataloader

    # evaluators
    cfg.val_evaluator = dict(type='AUC') #验证指标
    cfg.test_evaluator = cfg.val_evaluator
    return cfg

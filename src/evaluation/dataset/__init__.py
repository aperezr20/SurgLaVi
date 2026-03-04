import torch
import numpy
import random
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from dataset.workflow_dataset import WorkflowDataset, CholecT50Dataset


def get_media_type(dataset_config):
    return dataset_config[2]


def create_dataset(dataset_type, config):
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    normalize = transforms.Normalize(mean, std)
    type_transform = transforms.Lambda(lambda x: x.float().div(255.0))
    transform = transforms.Compose(
        [
            transforms.Resize(
                (config.inputs.image_res, config.inputs.image_res),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True
            ),
            type_transform,
            normalize,
        ]
    )
    video_only_dataset_kwargs_eval = dict(
        sample_type=config.inputs.video_input.sample_type_test,
        num_frames=config.inputs.video_input.num_frames_test,
    )
    test_datasets = []
    test_dataset_names = []

    # multiple test datasets, all separate
    data_file = config.test_file if "eval" in dataset_type else config.train_file
    
    for name, data_cfg in data_file.items():
        media_type = get_media_type(data_cfg)
        test_dataset_cls = (
                CholecT50Dataset if 'cholect50' in name else
                WorkflowDataset if media_type == "video" else
                None
            )
        test_dataset_names.append(name)
        dataset_kwargs = dict(
            config=config,
            ann_file=data_cfg,
            transform=transform,
        )
        if media_type == "video":
            dataset_kwargs.update(video_only_dataset_kwargs_eval)
        
        dataset = test_dataset_cls(**dataset_kwargs)
        test_datasets.append(dataset)

    return test_datasets, test_dataset_names


def worker_init_fn(worker_id):                                                                                                                                
    seed = 42              
                                                                                                                                   
    torch.manual_seed(seed)                                                                                                                                   
    torch.cuda.manual_seed(seed)                                                                                                                              
    torch.cuda.manual_seed_all(seed)                                                                                          
    numpy.random.seed(seed)                                                                                                             
    random.seed(seed)                                                                                                       
    torch.manual_seed(seed)                                                                                                                                   
    return


g = torch.Generator()
g.manual_seed(0)


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(
        datasets, samplers, batch_size, num_workers, is_trains, collate_fns
    ):
        if is_train:
            shuffle = sampler is None
            drop_last = False
        else:
            shuffle = False
            drop_last = False
        
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=False,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
            worker_init_fn=worker_init_fn,
            persistent_workers=True if n_worker > 0 else False,
        )
        loaders.append(loader)
    return loaders
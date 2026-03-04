import datetime
import logging
import time
from os.path import join
import pandas as pd
import torch
from tasks.workflow_utils import evaluation_wrapper
from utils.basic_utils import MetricLogger, SmoothedValue
from utils.config_utils import setup_main
from dataset import create_dataset, create_loader
import surgclip

logger = logging.getLogger(__name__)


def setup_dataloaders(config, mode="pt"):
    test_datasets, test_dataset_names = create_dataset(f"{mode}_eval", config)
    test_loaders = create_loader(
        test_datasets,
        [None] * len(test_datasets),
        batch_size=[config.inputs.batch_size_test[d.media_type] for d in test_datasets],
        num_workers=[config.num_workers] * len(test_datasets),
        is_trains=[False] * len(test_datasets),
        collate_fns=[None] * len(test_datasets),
    )
    return {k: v for k, v in zip(test_dataset_names, test_loaders)}


def main(config):
    device = torch.device(config.device)

    test_name2loaders = setup_dataloaders(config, mode=config.mode)
    model, preprocess, tokenizer = surgclip.load("SurgCLIP-B", device=device)

    logger.info("Start evaluating")
    start_time = time.time()
    eval_res = {}
    eval_results_path = join(config.output_dir, "results.json")

    with torch.cuda.amp.autocast(enabled=config.fp16):
        for test_name, test_loader in test_name2loaders.items():
            logger.info(f"Evaluating {test_name} dataset ------------")
            if test_name not in config.test_types:
                logger.info(f"Skip eval {test_name} split. All test_types {config.test_types}")
                continue
            res = evaluation_wrapper(model, test_loader, tokenizer, device, config, prefix=test_name)
            eval_res.update(res)
            pd.DataFrame(eval_res).to_json(eval_results_path)

    total_time = time.time() - start_time
    logger.info(f"Testing time {str(datetime.timedelta(seconds=int(total_time)))}")


if __name__ == "__main__":
    cfg = setup_main()
    main(cfg)
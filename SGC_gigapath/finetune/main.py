import os
import torch
import pandas as pd
import numpy as np
import torch.distributed as dist

from training import train
from params import get_finetune_params
from task_configs.utils import load_task_config
from utils import seed_torch, get_exp_code, get_splits, get_loader, save_obj
from datasets.slide_datatset import SlideDataset


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


if __name__ == '__main__':
    args = get_finetune_params()
    print(args)

    args.distributed, args.rank, args.world_size, args.local_rank = setup_distributed()
    args.is_main_process = (args.rank == 0)

    # set the device
    if args.distributed:
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # set the random seed
    seed_torch(device, args.seed + args.rank)

    # load the task configuration
    if args.is_main_process:
        print('Loading task configuration from: {}'.format(args.task_cfg_path))
    args.task_config = load_task_config(args.task_cfg_path)
    if args.is_main_process:
        print(args.task_config)
    args.task = args.task_config.get('name', 'task')
    
    # set the experiment save directory
    args.save_dir = os.path.join(args.save_dir, args.task, args.exp_name)
    args.model_code, args.task_code, args.exp_code = get_exp_code(args) # get the experiment code
    args.save_dir = os.path.join(args.save_dir, args.exp_code)
    if args.is_main_process:
        os.makedirs(args.save_dir, exist_ok=True)
        print('Experiment code: {}'.format(args.exp_code))
        print('Setting save directory: {}'.format(args.save_dir))

    # set the learning rate
    eff_batch_size = args.batch_size * args.gc * args.world_size
    if args.lr is None or args.lr < 0:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    if args.is_main_process:
        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)
        print("accumulate grad iterations: %d" % args.gc)
        print("effective batch size: %d" % eff_batch_size)

    # set the split key
    if args.pat_strat:
        args.split_key = 'pat_id'
    else:
        args.split_key = 'slide_id'

    # set up the dataset
    args.split_dir = os.path.join(args.split_dir, args.task_code) if not args.pre_split_dir else args.pre_split_dir
    if args.is_main_process:
        os.makedirs(args.split_dir, exist_ok=True)
        print('Setting split directory: {}'.format(args.split_dir))
    dataset = pd.read_csv(args.dataset_csv) # read the dataset csv file

    # use the slide dataset
    DatasetClass = SlideDataset

    # set up the results dictionary
    results = {}

    # start cross validation
    for fold in range(args.folds):
        # set up the fold directory
        save_dir = os.path.join(args.save_dir, f'fold_{fold}')
        if args.is_main_process:
            os.makedirs(save_dir, exist_ok=True)
        # get the splits
        train_splits, val_splits, test_splits = get_splits(dataset, fold=fold, **vars(args))
        # instantiate the dataset
        train_data, val_data, test_data = DatasetClass(dataset, args.root_path, train_splits, args.task_config, split_key=args.split_key) \
                                        , DatasetClass(dataset, args.root_path, val_splits, args.task_config, split_key=args.split_key) if len(val_splits) > 0 else None \
                                        , DatasetClass(dataset, args.root_path, test_splits, args.task_config, split_key=args.split_key) if len(test_splits) > 0 else None
        args.n_classes = train_data.n_classes # get the number of classes
        # Only rank 0 runs eval/test loaders to avoid duplicated logging/artifacts.
        if args.distributed and not args.is_main_process:
            val_data, test_data = None, None
        # get the dataloader
        train_loader, val_loader, test_loader = get_loader(train_data, val_data, test_data, **vars(args))
        # start training
        val_records, test_records = train((train_loader, val_loader, test_loader), fold, args)

        # update the results
        if args.is_main_process:
            records = {'val': val_records, 'test': test_records}
            for record_ in records:
                if records[record_] is None:
                    continue
                for key in records[record_]:
                    if 'prob' in key or 'label' in key:
                        continue
                    key_ = record_ + '_' + key
                    if key_ not in results:
                        results[key_] = []
                    results[key_].append(records[record_][key])

    if args.is_main_process:
        # save the results into a csv file
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(args.save_dir, 'summary.csv'), index=False)
        # print the results, mean and std
        for key in results_df.columns:
            print('{}: {:.4f} +- {:.4f}'.format(key, np.mean(results_df[key]), np.std(results_df[key])))
        print('Results saved in: {}'.format(os.path.join(args.save_dir, 'summary.csv')))
        print('Done!')

    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()

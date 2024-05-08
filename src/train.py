import os

import torch
import torch.multiprocessing as mp
from trainer.trainer import Trainer
from utils.option import args


def main_worker(id, ngpus_per_node, args):
    args.local_rank = args.global_rank = id
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        print(f"using GPU {args.world_size}-{args.global_rank} for training")
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=args.init_method,
            world_size=args.world_size,
            rank=args.global_rank,
            group_name="mtorch",
        )

    args.save_dir = os.path.join(args.save_dir, f"{args.model}_{args.data_train}_{args.mask_type}{args.image_size}")

    if (not args.distributed) or args.global_rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, "config.txt"), "a") as f:
            for key, val in vars(args).items():
                f.write(f"{key}: {val}\n")
        print(f"[**] create folder {args.save_dir}")

    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    torch.manual_seed(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # setup distributed parallel training environments
    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1:
        args.world_size = ngpus_per_node
        args.init_method = f"tcp://127.0.0.1:{args.port}"
        args.distributed = True
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        args.world_size = 1
        args.distributed = False
        main_worker(0, 1, args)

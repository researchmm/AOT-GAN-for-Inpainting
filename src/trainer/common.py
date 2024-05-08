import torch
from torch import distributed as dist


class timer:
    def __init__(self):
        self.acc = 0
        self.t0 = torch.cuda.Event(enable_timing=True)
        self.t1 = torch.cuda.Event(enable_timing=True)
        self.tic()

    def tic(self):
        self.t0.record()

    def toc(self, restart=False):
        self.t1.record()
        torch.cuda.synchronize()
        diff = self.t0.elapsed_time(self.t1) / 1000.0
        if restart:
            self.tic()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0


def reduce_loss_dict(loss_dict, world_size):
    if world_size == 1:
        return loss_dict

    with torch.no_grad():
        keys = []
        losses = []

        for k in sorted(loss_dict.keys()):
            keys.append(k)
            losses.append(loss_dict[k])

        losses = torch.stack(losses, 0)
        dist.reduce(losses, dst=0)

        if dist.get_rank() == 0:
            losses /= world_size

        reduced_losses = dict(zip(keys, losses))
    return reduced_losses

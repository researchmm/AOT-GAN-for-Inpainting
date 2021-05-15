import rasterio
import torch
from scipy.ndimage import zoom
from skimage.draw import rectangle_perimeter, line
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from scipy import stats
from glob import glob
import random


class Helper:
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def get_ranges(x):
        bmax = np.max(x.reshape(-1, x.shape[2] ** 2), axis=1)
        bmin = np.min(x.reshape(-1, x.shape[2] ** 2), axis=1)
        return bmax - bmin

    @staticmethod
    def get_tri(x):
        return np.apply_along_axis(Helper._get_tri, 1, x.reshape(-1, x.shape[2] ** 2))

    @staticmethod
    def _get_tri(x):
        now = x.reshape(-1, 30)
        tri = np.zeros_like(now)

        d = 1
        for i in range(1, 29):
            for j in range(1, 29):
                tri[i, j] = np.sqrt(
                    np.sum(
                        np.power(
                            now[i - d : i + d + 1, j - d : j + d + 1].flatten()
                            - now[i, j],
                            2,
                        )
                    )
                )
        return stats.trim_mean(tri.flatten(), 0.2)


class TerrainDataset(Dataset):
    NAN = 0

    def __init__(
        self,
        dataset_glob,
        dataset_type,
        patch_size=30,
        sample_size=256,
        observer_pad=50,
        block_variance=4,
        observer_height=0.75,
        limit_samples=None,
        randomize=True,
        random_state=42,
        usable_portion=1.0,
        fast_load=False,
        transform=None,
    ):
        """
        dataset_glob -> glob to *.tif files (i.e. "data/MDRS/data/*.tif")
        dataset_type -> train or validation
        patch_size -> the 1m^2 area to read from .TIF
        sample_size -> the 0.1m^2 res area to be trained sample size
        observer_pad -> n pixels to pad before getting a random observer
        block_variance -> how many different observer points
        observer_height -> Observer Height
        limit_samples -> Limit number of samples returned
        randomize -> predictable randomize
        random_state -> a value that gets added to seed
        usable_portion -> What % of the data will be used
        fast_load -> initialize from npy file, Warning: Dragons be aware
        transform -> if there is any, PyTorch Transforms
        """
        np.seterr(divide="ignore", invalid="ignore")

        # * Set Dataset attributes
        self.observer_height = observer_height
        self.patch_size = patch_size
        self.sample_size = sample_size
        self.block_variance = block_variance
        self.observer_pad = observer_pad

        # * PyTorch Related Variables
        self.transform = transform

        # * Gather files
        self.files = glob(dataset_glob)
        self.dataset_type = dataset_type
        self.usable_portion = usable_portion
        self.limit_samples = limit_samples

        self.randomize = False if fast_load else randomize
        self.random_state = random_state
        if self.randomize:
            random.shuffle(self.files)

        # * Build dataset dictionary
        self.sample_dict = dict()
        start = 0
        for file in tqdm(self.files, ncols=100, disable=fast_load):
            blocks, mask = self.get_blocks(file, return_mask=True)

            if len(blocks) == 0:
                continue

            self.sample_dict[file] = {
                "start": start,
                "end": start + len(blocks[mask]),
                "mask": mask,
                "min": np.min(blocks[mask]),
                "max": np.max(blocks[mask]),
                "range": np.max(Helper.get_ranges(blocks[mask])),
            }
            start += len(blocks[mask])

            del blocks
            if fast_load:
                break

        self.data_min = min(self.sample_dict.values(), key=lambda x: x["min"])["min"]
        self.data_max = max(self.sample_dict.values(), key=lambda x: x["max"])["max"]
        self.data_range = max(self.sample_dict.values(), key=lambda x: x["range"])[
            "range"
        ]

        # * Check if limit_samples is enough for this dataset
        if limit_samples is not None:
            assert (
                limit_samples <= self.get_len()
            ), "limit_samples cannot be bigger than dataset size"

        # * Dataset state
        self.current_file = None
        self.current_blocks = None
        self.idx_offset = 0

    def get_len(self):
        key = list(self.sample_dict.keys())[-1]
        return self.sample_dict[key]["end"]

    def __len__(self):
        if not self.limit_samples is None:
            return self.limit_samples
        return self.get_len()

    def __getitem__(self, idx):
        while True:
            target, mask, file_name = self.__internal_getitem__(idx + self.idx_offset)
            if not np.all(mask.numpy() == 0):
                break
            else:
                self.idx_offset += 1
                print(f"Skipped block #{self.idx_offset}")
        return target, mask, file_name

    def __internal_getitem__(self, idx):
        """
        returns (x, (ox, oy, oz)), y
        """
        rel_idx = None
        for file, info in self.sample_dict.items():
            if idx >= info["start"] and idx < info["end"]:
                rel_idx = idx - info["start"]
                if self.current_file != file:
                    b = self.get_blocks(file)
                    self.current_blocks = b[info["mask"]]
                    self.current_file = file
                break

        current = np.copy(self.current_blocks[rel_idx])
        current -= np.min(current)
        current /= self.data_range
        oh = self.observer_height / self.data_range

        adjusted = self.get_adjusted(current)
        viewshed, _ = self.viewshed(adjusted, oh, idx)
        mask = np.isnan(viewshed).astype(np.uint8)

        mask = torch.from_numpy(mask).float()
        mask = mask.unsqueeze(0)

        target = torch.from_numpy(adjusted).float()
        target = target.unsqueeze(0)

        return target, mask, f"{self.current_file}-{idx}"

    def viewshed(self, dem, oh, seed):
        h, w = dem.shape
        np.random.seed(seed + self.random_state)
        rands = np.random.rand(h - self.observer_pad, w - self.observer_pad)
        template = np.zeros_like(dem)
        template[
            self.observer_pad - self.observer_pad // 2 : h - self.observer_pad // 2,
            self.observer_pad - self.observer_pad // 2 : w - self.observer_pad // 2,
        ] = rands
        observer = tuple(np.argwhere(template == np.max(template))[0])

        yp, xp = observer
        zp = dem[observer] + oh
        observer = (xp, yp, zp)
        viewshed = np.copy(dem)

        # * Find perimiter
        rr, cc = rectangle_perimeter((1, 1), end=(h - 2, w - 2), shape=dem.shape)

        # * Iterate through perimiter
        for yc, xc in zip(rr, cc):
            # * Form the line
            ray_y, ray_x = line(yp, xp, yc, xc)
            ray_z = dem[ray_y, ray_x]

            m = (ray_z - zp) / np.hypot(ray_y - yp, ray_x - xp)

            max_so_far = -np.inf
            for yi, xi, mi in zip(ray_y, ray_x, m):
                if mi < max_so_far:
                    viewshed[yi, xi] = np.nan
                else:
                    max_so_far = mi

        return viewshed, observer

    def blockshaped(self, arr, nside):
        """
        Return an array of shape (n, nside, nside) where
        n * nside * nside = arr.size

        If arr is a 2D array, the returned array should look like n subblocks with
        each subblock preserving the "physical" layout of arr.
        """
        h, w = arr.shape
        assert h % nside == 0, "{} rows is not evenly divisble by {}".format(h, nside)
        assert w % nside == 0, "{} cols is not evenly divisble by {}".format(w, nside)
        return (
            arr.reshape(h // nside, nside, -1, nside)
            .swapaxes(1, 2)
            .reshape(-1, nside, nside)
        )

    def get_adjusted(self, block):
        zoomed = zoom(block, 10, order=1)
        y, x = zoomed.shape
        startx = x // 2 - (self.sample_size // 2)
        starty = y // 2 - (self.sample_size // 2)
        return zoomed[
            starty : starty + self.sample_size, startx : startx + self.sample_size
        ]

    def get_blocks(self, file, return_mask=False):
        raster = rasterio.open(file)
        grid = raster.read(1)

        # Remove minimum
        grid[grid == np.min(grid)] = np.nan

        # Find the edges to cut from
        NL = np.count_nonzero(np.isnan(grid[:, 0]))
        NR = np.count_nonzero(np.isnan(grid[:, -1]))
        NT = np.count_nonzero(np.isnan(grid[0, :]))
        NB = np.count_nonzero(np.isnan(grid[-1, :]))

        w, h = grid.shape
        if NL > NR:
            grid = grid[w % self.patch_size : w, 0:h]
        else:
            grid = grid[0 : w - (w % self.patch_size), 0:h]

        w, h = grid.shape
        if NT > NB:
            grid = grid[0:w, h % self.patch_size : h]
        else:
            grid = grid[0:w, 0 : h - (h % self.patch_size)]

        blocks = self.blockshaped(grid, self.patch_size)

        # * Randomize
        if self.randomize:
            np.random.seed(self.random_state)
            np.random.shuffle(blocks)

        # * Remove blocks that contain nans
        mask = ~np.isnan(blocks).any(axis=1).any(axis=1)
        blocks = blocks[mask]

        if self.dataset_type == "train":
            blocks = blocks[: int(len(blocks) * self.usable_portion)]
        else:
            blocks = blocks[int(len(blocks) * self.usable_portion) :]

        # * Add Variance
        blocks = np.repeat(blocks, self.block_variance, axis=0)

        if return_mask:
            # * Further filter remeaning data in relation to z-score
            ranges = Helper.get_ranges(blocks)
            mask_ru = np.abs(stats.zscore(ranges)) < 2
            mask_rl = np.abs(stats.zscore(ranges)) > 0.2

            # # * Terrain Ruggedness Index
            # tri = Helper.get_tri(blocks)
            # mask_tu = np.abs(stats.zscore(tri)) < 2
            # mask_tl = np.abs(stats.zscore(tri)) > 0.05

            mask = mask_ru & mask_rl  # & mask_tu & mask_tl
            return blocks, mask
        return blocks

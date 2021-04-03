import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image
from multiprocessing import Pool

from metric import metric as module_metric

parser = argparse.ArgumentParser(description='Image Inpainting')
parser.add_argument('--real_dir', required=True, type=str)
parser.add_argument('--fake_dir', required=True, type=str)
parser.add_argument("--metric", type=str, nargs="+")
args = parser.parse_args()


def read_img(name_pair): 
    rname, fname = name_pair
    rimg = Image.open(rname)
    fimg = Image.open(fname)
    return np.array(rimg), np.array(fimg)


def main(num_worker=8):

    real_names = sorted(list(glob(f'{args.real_dir}/*.png')))
    fake_names = sorted(list(glob(f'{args.fake_dir}/*.png')))
    print(f'real images: {len(real_names)}, fake images: {len(fake_names)}')
    real_images = []
    fake_images = []
    pool = Pool(num_worker)
    for rimg, fimg in tqdm(pool.imap_unordered(read_img, zip(real_names, fake_names)), total=len(real_names), desc='loading images'):
        real_images.append(rimg)
        fake_images.append(fimg)


    # metrics prepare for image assesments
    metrics = {met: getattr(module_metric, met) for met in args.metric}
    evaluation_scores = {key: 0 for key,val in metrics.items()}
    for key, val in metrics.items():
        evaluation_scores[key] = val(real_images, fake_images, num_worker=num_worker)
    print(' '.join(['{}: {:6f},'.format(key, val) for key,val in evaluation_scores.items()]))
  
  


if __name__ == '__main__':
    main()
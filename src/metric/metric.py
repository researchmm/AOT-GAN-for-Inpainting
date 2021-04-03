import os 
import pickle
import numpy as np
from tqdm import tqdm
from scipy import linalg
from multiprocessing import Pool
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

import torch
from torch.autograd import Variable
from torch.nn.functional import adaptive_avg_pool2d

from .inception import InceptionV3



# ============================

def compare_mae(pairs):
    real, fake = pairs
    real, fake = real.astype(np.float32), fake.astype(np.float32)
    return np.sum(np.abs(real - fake)) / np.sum(real + fake)

def compare_psnr(pairs):
    real, fake = pairs
    return peak_signal_noise_ratio(real, fake)

def compare_ssim(pairs):
    real, fake = pairs
    return structural_similarity(real, fake, multichannel=True)

# ================================

def mae(reals, fakes, num_worker=8):
    error = 0
    pool = Pool(num_worker)
    for val in tqdm(pool.imap_unordered(compare_mae, zip(reals, fakes)), total=len(reals), desc='compare_mae'):
        error += val 
    return error / len(reals)

def psnr(reals, fakes, num_worker=8):
    error = 0
    pool = Pool(num_worker)
    for val in tqdm(pool.imap_unordered(compare_psnr, zip(reals, fakes)), total=len(reals), desc='compare_psnr'):
        error += val
    return error / len(reals)

def ssim(reals, fakes, num_worker=8):
    error = 0
    pool = Pool(num_worker)
    for val in tqdm(pool.imap_unordered(compare_ssim, zip(reals, fakes)), total=len(reals), desc='compare_ssim'):
        error += val
    return error / len(reals)

def fid(reals, fakes, num_worker=8, real_fid_path=None):
    
    dims = 2048
    batch_size = 4
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).cuda()

    if real_fid_path is None: 
        real_fid_path = 'places2_fid.pt'
        
    if os.path.isfile(real_fid_path): 
        data = pickle.load(open(real_fid_path, 'rb'))
        real_m, real_s = data['mu'], data['sigma']
    else: 
        reals = (np.array(reals).astype(np.float32) / 255.0).transpose((0, 3, 1, 2))
        real_m, real_s = calculate_activation_statistics(reals, model, batch_size, dims)
        with open(real_fid_path, 'wb') as f: 
            pickle.dump({'mu': real_m, 'sigma': real_s}, f)


    # calculate fid statistics for fake images
    fakes = (np.array(fakes).astype(np.float32) / 255.0).transpose((0, 3, 1, 2))
    fake_m, fake_s = calculate_activation_statistics(fakes, model, batch_size, dims)

    fid_value = calculate_frechet_distance(real_m, real_s, fake_m, fake_s)

    return fid_value


def calculate_activation_statistics(images, model, batch_size=64,
                                    dims=2048, cuda=True, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                     must lie between 0 and 1.
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(images, model, batch_size, dims, cuda, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def get_activations(images, model, batch_size=64, dims=2048, cuda=True, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                     must lie between 0 and 1.
    -- model       : Instance of inception model
    -- batch_size  : the images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size depends
                     on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    d0 = images.shape[0]
    if batch_size > d0:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = d0

    n_batches = d0 // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, dims))
    for i in tqdm(range(n_batches), desc='calculate activations'):
        if verbose:
            print('\rPropagating batch %d/%d' %
                  (i + 1, n_batches), end='', flush=True)
        start = i * batch_size
        end = start + batch_size

        batch = torch.from_numpy(images[start:end]).type(torch.FloatTensor)
        batch = Variable(batch)
        if torch.cuda.is_available:
            batch = batch.cuda()
        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)
    if verbose:
        print(' done')

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an 
               representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an 
               representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'
    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

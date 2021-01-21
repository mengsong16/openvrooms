import torch
import numpy as np
import os
import random



def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path



def batch_from_obs(obs, batch_size=32):
	"""Converts a pixel obs (C,H,W) to a batch (B,C,H,W) of given size"""
	if isinstance(obs, torch.Tensor):
		if len(obs.shape)==3:
			obs = obs.unsqueeze(0)
		return obs.repeat(batch_size, 1, 1, 1)

	if len(obs.shape)==3:
		obs = np.expand_dims(obs, axis=0)
	return np.repeat(obs, repeats=batch_size, axis=0)


def _rotate_single_with_label(x, label):
	"""Rotate an image"""
	if label == 1:
		return x.flip(2).transpose(1, 2)
	elif label == 2:
		return x.flip(2).flip(1)
	elif label == 3:
		return x.transpose(1, 2).flip(2)
	return x


def rotate(x):
	"""Randomly rotate a batch of images and return labels"""
	images = []
	labels = torch.randint(4, (x.size(0),), dtype=torch.long).to(x.device)
	for img, label in zip(x, labels):
		img = _rotate_single_with_label(img, label)
		images.append(img.unsqueeze(0))

	return torch.cat(images), labels


def random_crop_cuda(x, size=84, w1=None, h1=None, return_w1_h1=False):
	"""Vectorized CUDA implementation of random crop"""
	assert isinstance(x, torch.Tensor) and x.is_cuda, \
		'input must be CUDA tensor'
	
	n = x.shape[0]
	img_size = x.shape[-1]
	crop_max = img_size - size

	if crop_max <= 0:
		if return_w1_h1:
			return x, None, None
		return x

	x = x.permute(0, 2, 3, 1)

	if w1 is None:
		w1 = torch.LongTensor(n).random_(0, crop_max)
		h1 = torch.LongTensor(n).random_(0, crop_max)

	windows = view_as_windows_cuda(x, (1, size, size, 1))[..., 0,:,:, 0]
	cropped = windows[torch.arange(n), w1, h1]

	if return_w1_h1:
		return cropped, w1, h1

	return cropped


def view_as_windows_cuda(x, window_shape):
	"""PyTorch CUDA-enabled implementation of view_as_windows"""
	assert isinstance(window_shape, tuple) and len(window_shape) == len(x.shape), \
		'window_shape must be a tuple with same number of dimensions as x'
	
	slices = tuple(slice(None, None, st) for st in torch.ones(4).long())
	win_indices_shape = [
		x.size(0),
		x.size(1)-int(window_shape[1]),
		x.size(2)-int(window_shape[2]),
		x.size(3)    
	]

	new_shape = tuple(list(win_indices_shape) + list(window_shape))
	strides = tuple(list(x[slices].stride()) + list(x.stride()))

	return x.as_strided(new_shape, strides)


def random_crop(imgs, size=84, w1=None, h1=None, return_w1_h1=False):
	"""Vectorized random crop, imgs: (B,C,H,W), size: output size"""
	assert (w1 is None and h1 is None) or (w1 is not None and h1 is not None), \
		'must either specify both w1 and h1 or neither of them'

	is_tensor = isinstance(imgs, torch.Tensor)
	if is_tensor:
		assert imgs.is_cuda, 'input images are tensors but not cuda!'
		return random_crop_cuda(imgs, size=size, w1=w1, h1=h1, return_w1_h1=return_w1_h1)
		
	n = imgs.shape[0]
	img_size = imgs.shape[-1]
	crop_max = img_size - size

	if crop_max <= 0:
		if return_w1_h1:
			return imgs, None, None
		return imgs

	imgs = np.transpose(imgs, (0, 2, 3, 1))
	if w1 is None:
		w1 = np.random.randint(0, crop_max, n)
		h1 = np.random.randint(0, crop_max, n)

	windows = view_as_windows(imgs, (1, size, size, 1))[..., 0,:,:, 0]
	cropped = windows[np.arange(n), w1, h1]

	if return_w1_h1:
		return cropped, w1, h1

	return cropped
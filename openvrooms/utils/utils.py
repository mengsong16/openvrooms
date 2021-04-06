import os
import numpy as np
import yaml
import collections
from scipy.spatial.transform import Rotation as R
from transforms3d import quaternions
from packaging import version
from transforms3d.euler import euler2quat, quat2euler
from gibson2.utils.utils import quatToXYZW, quatFromXYZW


# The function to retrieve the rotation matrix changed from as_dcm to as_matrix in version 1.4
# We will use the version number for backcompatibility
import scipy
scipy_version = version.parse(scipy.version.version)

# File I/O related

# constants
GRAVITY = 9.8
# quat: [w,x,y,z], n1*n2*...*4
def quat_identity():
	return np.array([1, 0, 0, 0])


# quat: [w,x,y,z], n1*n2*...*4
def quat_difference(q, p):
	return quat_normalize(quat_mul(q, quat_conjugate(p)))

# quat: [w,x,y,z], n1*n2*...*4
def quat_magnitude(q):
	w = q[..., 0]
	assert np.all(w >= 0)
	return 2 * np.arccos(np.clip(w, -1.0, 1.0))

# quat: [w,x,y,z], n1*n2*...*4
def quat_normalize(q):
	assert q.shape[-1] == 4
	# -1, 0, +1
	sign = np.sign(q[..., [0]])
	# Sign takes value of 0 whenever the input is 0, but we actually don't want to do that
	sign[sign == 0] = 1
	# -1 --> 1, 1-->1, 0-->0?
	return q * sign  # use quat with w >= 0

def l2_distance(v1, v2):
	"""Returns the L2 distance between vector v1 and v2."""
	return np.linalg.norm(np.array(v1) - np.array(v2))

def normalize_angles(angles, low=-np.pi, high=np.pi):
	"""Puts angles in [low, high] range."""
	angles = angles.copy()
	if angles.size > 0:
		angles = np.mod(angles - low, high - low) + low
		assert low - 1e-6 <= angles.min() and angles.max() <= high + 1e-6
	return angles
	
def subtract_euler(e1, e2):
	assert e1.shape == e2.shape
	assert e1.shape[-1] == 3
	q1 = euler2quat_array(e1)
	q2 = euler2quat_array(e2)
	q_diff = subtract_quat(q1, q2)
	return quat2euler_array(q_diff) 

# quat: [w,x,y,z]
def subtract_quat(q1, q2):
	q_diff = quat_mul(q1, quat_conjugate(q2))
	return q_diff

# quat: [w,x,y,z], n1*n2*...*4
def quat_conjugate(q):
	inv_q = -q
	inv_q[..., 0] *= -1
	return inv_q

# quat: [w,x,y,z], n1*n2*...*4
def quat_mul(q0, q1):
	assert q0.shape == q1.shape
	assert q0.shape[-1] == 4
	assert q1.shape[-1] == 4

	w0 = q0[..., 0]
	x0 = q0[..., 1]
	y0 = q0[..., 2]
	z0 = q0[..., 3]

	w1 = q1[..., 0]
	x1 = q1[..., 1]
	y1 = q1[..., 2]
	z1 = q1[..., 3]

	w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
	x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
	y = w0 * y1 + y0 * w1 + z0 * x1 - x0 * z1
	z = w0 * z1 + z0 * w1 + x0 * y1 - y0 * x1
	q = np.array([w, x, y, z])
	if q.ndim == 2:
		q = q.swapaxes(0, 1)
	assert q.shape == q0.shape
	return q

# orn: np.arraylike
def quatFromXYZW_array(orn, seq):
	orn = np.array(orn)
	if orn.ndim >= 2:
		orn_new = np.reshape(orn, (-1, orn.shape[-1]))
		n = range(orn_new.shape[0])
		res = [quatFromXYZW(orn_new[i], seq) for i in list(n)]
		res = np.reshape(np.array(res), orn.shape)
		return res
	else:
		return quatFromXYZW(orn, seq)

# orn: np.arraylike
def	quatToXYZW_array(orn, seq):
	orn = np.array(orn)
	if orn.ndim >= 2:
		orn_new = np.reshape(orn, (-1, orn.shape[-1]))
		n = range(orn_new.shape[0])
		res = [quatToXYZW(orn_new[i], seq) for i in list(n)]
		res = np.reshape(np.array(res), orn.shape)
		return res
	else:
		return quatToXYZW(orn, seq)


# orn: np.arraylike
# quat: w,x,y,z
# euler: in radian
# default axes='sxyz'
def	euler2quat_array(orn):
	orn = np.array(orn)
	if orn.ndim >= 2:
		orn_new = np.reshape(orn, (-1, orn.shape[-1]))
		n = range(orn_new.shape[0])
		res = [euler2quat(orn_new[i][0], orn_new[i][1], orn_new[i][2]) for i in list(n)]
		res_shape = np.array(orn.shape)
		res_shape[-1] = 4
		res = np.reshape(np.array(res), res_shape)
		return res
	else:
		return euler2quat(orn)


# orn: np.arraylike
# quat: w,x,y,z
# euler: in radian
# default axes='sxyz'
def	quat2euler_array(orn):
	orn = np.array(orn)
	if orn.ndim >= 2:
		orn_new = np.reshape(orn, (-1, orn.shape[-1]))
		n = range(orn_new.shape[0])
		res = [quat2euler(orn_new[i]) for i in list(n)]
		res_shape = np.array(orn.shape)
		res_shape[-1] = 3
		res = np.reshape(np.array(res), res_shape)
		return res
	else:
		return quat2euler(orn)	


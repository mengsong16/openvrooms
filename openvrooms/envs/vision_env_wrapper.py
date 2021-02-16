import numpy as np
from numpy.random import randint
import os
import gym
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import cv2
from collections import deque
import imageio
import subprocess as sp
from skimage import img_as_ubyte

class FrameStack(gym.Wrapper):
	"""Stack frames as observation"""
	# C,H,W
	def __init__(self, env, k):
		gym.Wrapper.__init__(self, env)

		self._k = k
		self._frames = deque([], maxlen=k)
		shp = self.env.observation_space.shape
		low = np.amin(self.env.observation_space.low)
		high = np.amax(self.env.observation_space.high)

		if self.env.image_shape == "HWC":
			self.observation_space = gym.spaces.Box(
				low=low,
				high=high,
				shape=(shp[0], shp[1], shp[2]*k),
				dtype=self.env.observation_space.dtype
			)
		else: #CHW
			self.observation_space = gym.spaces.Box(
				low=low,
				high=high,
				shape=((shp[0] * k,) + shp[1:]),
				dtype=self.env.observation_space.dtype
			)	
		#self._max_episode_steps = env._max_episode_steps

	def reset(self):
		obs = self.env.reset()
		# when reset, clear deque, repeat obs for k times
		for _ in range(self._k):
			self._frames.append(obs)
		return self._get_obs()

	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		self._frames.append(obs)
		return self._get_obs(), reward, done, info

	# return numpy array
	def _get_obs(self):
		assert len(self._frames) == self._k
		if self.env.image_shape == "HWC":
			return np.concatenate(list(self._frames), axis=2)
		else:
			return np.concatenate(list(self._frames), axis=0)	
	

class VideoRecorder(gym.Wrapper):
	def __init__(self, env, dir_name="", file_name='video', file_format='gif', fps=25, width=128, height=128):
		gym.Wrapper.__init__(self, env)
		if not os.path.exists(dir_name):
			os.makedirs(dir_name)
		self.path = os.path.join(dir_name, file_name+'.'+file_format)
		self.fps = fps
		self.file_format = file_format
		self.frames = []
		self.enabled = False

		# need to install ffmpeg
		if self.file_format == 'mp4':
			self.command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "{}X{}".format(width, height),
						"-pix_fmt", "rgb24", "-r", str(fps), "-i", "-", "-an", "-vcodec", "mpeg4", self.path]
			try:
				self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
			except FileNotFoundError:
				print('Video recorder: Failed to build pipe')
				pass

	def start_video_recorder(self, clear=False):
		self.enabled = True
		if clear:
			self.frames = []

	def reset(self):
		obs = self.env.reset()

		if self.enabled:
			# H,W,C
			frame = self.env.render(mode='rgb')
			self.frames.append(frame)

		return obs

	def step(self, action):
		obs, reward, done, info = self.env.step(action)

		if self.enabled:
			# H,W,C
			frame = self.env.render(mode='rgb')
			self.frames.append(frame)
		
		return obs, reward, done, info

	def stop_video_recorder(self):
		self.enabled = False

	# gif or mp4
	def save(self):
		
		if self.file_format == 'mp4':
			frames = np.array(self.frames)
			# [0,1,float] --> [0,255,uint8]
			frames = np.array(frames*255, dtype='uint8')
			#print(frames.shape)
			self.pipe.stdin.write(frames.tostring())
		else:	
			imageio.mimsave(self.path, img_as_ubyte(self.frames), fps=self.fps)
		

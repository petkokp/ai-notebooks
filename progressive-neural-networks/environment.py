from skimage import util
from gymnasium.spaces.box import Box
import gymnasium as gym
from statistics import mean, median
import numpy as np
import cv2


def gen_stats(data):
    return min(data), max(data), mean(data), median(data)


def get_threshold(env_name):
    if 'Pong' in env_name:
        return 18
    elif 'Riverraid' in env_name:
        return 14000
    elif 'Boxing' in env_name:
        return 90
    elif 'NameThisGame' in env_name:
        return 14000
    elif 'Alien' in env_name:
        return 2500
    else:
        return 0


def cv2_clipped_zoom(ori, factor):
    h, w = ori.shape[:2]
    h_new, w_new = int(h * factor), int(w * factor)

    y1, x1 = max(0, h_new - h) // 2, max(0, w_new - w) // 2
    y2, x2 = y1 + h, x1 + w
    bbox = np.array([y1, x1, y2, x2])
    bbox = (bbox / factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    crop = ori[y1:y2, x1:x2]

    h_resize, w_resize = min(h_new, h), min(w_new, w)
    h1_pad, w1_pad = (h - h_resize) // 2, (w - w_resize) // 2
    h2_pad, w2_pad = (h - h_resize) - h1_pad, (w - w_resize) - w1_pad
    spec = [(h1_pad, h2_pad), (w1_pad, w2_pad)] + [(0, 0)] * (ori.ndim - 2)

    resize = cv2.resize(crop, (w_resize, h_resize))
    resize = np.pad(resize, spec, mode='constant')
    assert resize.shape[0] == h and resize.shape[1] == w
    return resize


def create_env(opt):
    envs = []
    if opt.envs == 'pong':
        if opt.ncolumns == 1:
            envs = [
                NormalizedEnv(AtariRescale(gym.make('PongDeterministic-v4')))
            ]
        elif opt.ncolumns == 2:
            envs = [
                NormalizedEnv(
                    PongNoisy(AtariRescale(gym.make('PongDeterministic-v4')))),
                NormalizedEnv(AtariRescale(gym.make('PongDeterministic-v4')))
            ]
        elif opt.ncolumns == 3:
            envs = [
                NormalizedEnv(
                    PongNoisy(AtariRescale(gym.make('PongDeterministic-v4')))),
                NormalizedEnv(AtariRescale(gym.make('PongDeterministic-v4'))),
                NormalizedEnv(
                    AtariRescale(PongHFlip(gym.make('PongDeterministic-v4'))))
            ]
    elif opt.envs == 'atari':
        if opt.ncolumns == 1:
            envs = [
                NormalizedEnv(AtariRescale(gym.make('BoxingDeterministic-v4')))
            ]
        elif opt.ncolumns == 2:
            envs = [
                NormalizedEnv(AtariRescale(gym.make('PongDeterministic-v4'))),
                NormalizedEnv(AtariRescale(gym.make('BoxingDeterministic-v4')))
            ]
        elif opt.ncolumns == 3:
            envs = [
                NormalizedEnv(AtariRescale(gym.make('PongDeterministic-v4'))),
                NormalizedEnv(
                    AtariRescale(gym.make('RiverraidDeterministic-v4'))),
                NormalizedEnv(AtariRescale(gym.make('BoxingDeterministic-v4')))
            ]
    return envs


class AtariRescale(gym.ObservationWrapper):
    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = Box(0.0, 1.0, [1, 84, 84])

    def observation(self, frame):
        frame = frame[34:34 + 160, :160]
        frame = cv2.resize(frame, (84, 84))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
        frame = frame.astype(np.float32)
        frame *= (1.0 / 255.0)
        frame = np.moveaxis(frame, -1, 0)
        return frame


class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.nsteps = 0

    def observation(self, observation):
        self.nsteps += 1
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.nsteps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.nsteps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)


class PongHFlip(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.unwrapped.spec.id = 'PongHFlipDeterministic-v4'

    def reset(self):
        observation = self.env.reset()
        observation = cv2.flip(observation, 1)
        return observation

    def step(self, action):
        if action == 2 or action == 4:
            action = 3
        elif action == 3 or action == 5:
            action = 4
        observation, reward, done, info = self.env.step(action)
        observation = cv2.flip(observation, 1)
        return observation, reward, done, info

    def render(self):
        self.env.render()


class PongNoisy(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.unwrapped.spec.id = 'PongNoisyDeterministic-v4'

    def reset(self):
        observation = self.env.reset()
        observation = util.random_noise(observation, mode='gaussian', seed=1)
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = util.random_noise(observation, mode='gaussian', seed=1)
        return observation, reward, done, info

    def render(self):
        self.env.render()


class PongZoom(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.unwrapped.spec.id = 'PongZoomDeterministic-v4'

    def reset(self):
        observation = self.env.reset()
        observation = cv2_clipped_zoom(observation, 0.75)
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = cv2_clipped_zoom(observation, 0.75)
        return observation, reward, done, info

    def render(self):
        self.env.render()

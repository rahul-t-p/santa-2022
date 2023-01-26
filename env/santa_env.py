import gym
from gym import spaces
import numpy as np
from functools import reduce
import cv2

import torch
from torchvision import transforms

import matplotlib.pyplot as plt


class Santa2k22(gym.Env):
    def __init__(self, resource_dir:str, do_render:bool = False):
        super(Santa2k22, self).__init__()

        self.state_space = cv2.imread(resource_dir+"/image.png")
        self.side, _, _ = self.state_space.shape
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.side, self.side),
                                     dtype=np.float32)

        self.key_to_links = dict({
            i : 2 ** (6-i) for i in range(8)
        })
        self.key_to_links[7] = 1
        self.action_space = spaces.Box(low=0, high=2, shape=(8,), dtype=int) # (links x direction) links; 0 => 64, 1 => 32 .... 7 => 1; direction; anticlock => 2, clock => 0, same position => 1

        self.state = None
        self.config = None
        self.done = False
        self.do_render = do_render
        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])

    def reset(self):
        self.state = torch.zeros((self.side, self.side), dtype=torch.float32) - 1
        self.state[self.side//2+1, self.side//2+1] = 1

        self.config = [(64, 0), (-32, 0), (-16, 0), (-8, 0), (-4, 0), (-2, 0), (-1, 0), (-1, 0)]

        return self.state[None, ...]

    def _get_position(self, config:list):
        return reduce(lambda p, q: (p[0] + q[0], p[1] + q[1]), config, (0, 0))

    def _rotate_link(self, vector, direction):
        x, y = vector
        if direction == 2:  # counter-clockwise
            if y >= x and y > -x:
                x -= 1
            elif y > x and y <= -x:
                y -= 1
            elif y <= x and y < -x:
                x += 1
            else:
                y += 1
        elif direction == 0:  # clockwise
            if y > x and y >= -x:
                x += 1
            elif y >= x and y < -x:
                y += 1
            elif y < x and y <= -x:
                x -= 1
            else:
                y -= 1
        return (x, y)


    def _rotate(self, config, i, direction):
        config = config.copy()
        config[i] = self._rotate_link(config[i], direction)
        return config

    def _cartesian_to_array(self, x, y, shape):
        m, n = shape[:2]
        i = (n - 1) // 2 - y
        j = (n - 1) // 2 + x
        if i < 0 or i >= m or j < 0 or j >= n:
            raise ValueError("Coordinates not within given dimensions.")
        return i, j

    def _reconfiguration_cost(self, from_config, to_config):
        nlinks = len(from_config)
        diffs = np.abs(np.asarray(from_config) - np.asarray(to_config)).sum(axis=1)
        return np.sqrt(diffs.sum())

    def _color_cost(self, from_position, to_position, color_scale=3.0):
        return np.abs(self.state_space[to_position] - self.state_space[from_position]).sum() * color_scale

    def _step_cost(self, from_config, to_config):
        from_position = self._cartesian_to_array(*self._get_position(from_config), self.state_space.shape)
        to_position = self._cartesian_to_array(*self._get_position(to_config), self.state_space.shape)
        return (
            self._reconfiguration_cost(from_config, to_config) +
            self._color_cost(from_position, to_position)
        )

    def step(self, action):

        prev_config = self.config.copy()
        prev_pos = self._get_position(self.config)
        prev_pos_array = self._cartesian_to_array(*prev_pos, (self.side, self.side, 3))

        for idx, direction in enumerate(action):
            self.config = self._rotate(self.config, idx, direction)
        new_pos = self._get_position(self.config)
        new_pos_array = self._cartesian_to_array(*new_pos, (self.side, self.side, 3))

        if(self.state[new_pos_array] != 1):
            self.state[new_pos_array] = 1

        reward = -1 * self._step_cost(prev_config, self.config)

        if torch.all(self.state.eq(torch.ones((self.side, self.side), dtype=torch.float32))):
            self.done = True

        if self.render:
            self.render()

        return self.state[None, ...], reward, self.done, {}

    def render(self, mode = "human"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""

        if mode == "human":
            state = np.zeros((self.side, self.side), dtype = np.uint8)
            state[self.state.numpy() == -1] = 0
            state[self.state.numpy() == 1] = 255
            plt.title("Christmas Card")
            plt.imshow(state)
        elif mode == "rgb_array":
            return self.state

    def random_action_test(self, step:int=10):
        self.reset()
        for i in range(step):
            action = self.action_space.sample()
            next_state, reward, done, _ = self.step(action)

        self.render()

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
from gym import utils
from rllab.envs.mujoco import mujoco_env

import tensorflow as tf

class CartpoleEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    PENDULUM_LENGTH = 0.6

    def __init__(self,
            ctrl_cost_coeff=1e-2,
            *args, **kwargs):
        self.viewer = None
        utils.EzPickle.__init__(self)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # mujoco_env.MujocoEnv.__init__(self, '%s/assets/cartpole.xml' % dir_path, 2)
        super(CartpoleEnv, self).__init__(*args, **kwargs, file_path='/home/azahed/me-trpo/vendor/mujoco_models/cartpole.xml')
        self.reset_model()

    def _step(self, a):
        # self.do_simulation(a, self.frame_skip)
        self.forward_dynamics(a)
        ob = self._get_obs()

        cost_lscale = CartpoleEnv.PENDULUM_LENGTH
        # reward = np.exp(
        #     -np.sum(np.square(self._get_ee_pos(ob) - np.array([0.0, CartpoleEnv.PENDULUM_LENGTH]))) / (cost_lscale ** 2)
        # )
        # reward -= 0.01 * np.sum(np.square(a))
        reward = -1 * self.cost_np(None, a, ob)

        done = False
        return ob, reward, done, {}

    def step(self, a):
        return self._step(a)

    @property
    def n_states(self):
        '''
        :return: state dimensions
        '''
        return 4

    @property
    def n_states(self):
        '''
        :return: state dimensions
        '''
        return 2
    def reset_model(self, init_state=None):
        if init_state is None:
            qpos = self.init_qpos.copy() + np.random.normal(0, 0.1, np.shape(self.init_qpos))
            qvel = self.init_qvel.copy() + np.random.normal(0, 0.1, np.shape(self.init_qvel))
            init_state = np.vstack([qpos, qvel, np.zeros(shape=(17,1))])
        else:
            qpos = init_state[:2]
            qvel = init_state[2:]
        # self.set_state(qpos, qvel)
        self.reset_mujoco(init_state)
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        obs = self._get_obs()
        return obs

    def reset(self, init_state=None):
        return self.reset_model(init_state)

    def _get_obs(self):
        return np.concatenate([self.model.data.qpos, self.model.data.qvel]).ravel()

    def get_current_obs(self):
        return self._get_obs()

    def get_obs(self):
        return self._get_obs()

    @staticmethod
    def _get_ee_pos(x):
        x0, theta = x[0], x[1]
        return np.array([
            x0 - CartpoleEnv.PENDULUM_LENGTH * np.sin(theta),
            -CartpoleEnv.PENDULUM_LENGTH * np.cos(theta)
        ])

    @staticmethod
    def _get_ee_pos_tf(x):
        x0, theta = x[:,0], x[:,1]
        print(x0)
        print(theta)
        return tf.concat([
            tf.expand_dims(x0 - CartpoleEnv.PENDULUM_LENGTH * tf.sin(theta),1),
            tf.expand_dims(-CartpoleEnv.PENDULUM_LENGTH * tf.cos(theta),1)
        ], axis=1)

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = v.model.stat.extent

    def cost_np(self, x, u, x_next, ctrl_cost_coeff=.1):
        if len(x_next.shape) == 2:
            return np.mean(self.cost_np_vec(x, u, x_next, ctrl_cost_coeff))
        cost = -1 * np.exp(
            -np.sum(np.square(self._get_ee_pos(x_next) - np.array([0.0, CartpoleEnv.PENDULUM_LENGTH]))) / (cost_lscale ** 2)
        )
        cost += 0.01 * np.sum(np.square(u))
        return cost


    def cost_tf(self, x, u, x_next, ctrl_cost_coeff=.1):
        cost = -1 * tf.exp(
            -tf.reduce_sum(np.square(self._get_ee_pos_tf(x_next) - np.array([0.0, CartpoleEnv.PENDULUM_LENGTH])), axis=1) / (cost_lscale ** 2)
        )
        cost += 0.01 * tf.reduce_sum(tf.square(u), axis=1)
        return cost

    def cost_np_vec(self, x, u, x_next, ctrl_cost_coeff=.1):
        
        cost = np.array([self.cost_np(None, u[i], x_next[i], ctrl_cost_coeff) for i in range(x_next.shape[0])])
        return cost
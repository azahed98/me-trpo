from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
from gym import utils
from rllab.envs.mujoco import mujoco_env


class PusherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
            ctrl_cost_coeff=1e-2,
            *args, **kwargs):
        self.viewer = None
        utils.EzPickle.__init__(self)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # mujoco_env.MujocoEnv.__init__(self, '%s/assets/pusher.xml' % dir_path, 4)
        super(PusherEnv, self).__init__(*args, **kwargs, file_path='/home/azahed/me-trpo/vendor/mujoco_models/pusher.xml')
        self.reset_model()

    @property
    def n_states(self):
        '''
        :return: state dimensions
        '''
        return 17


    def _step(self, a):
        # obj_pos = self.get_body_com("object")
        # vec_1 = obj_pos - self.get_body_com("tips_arm")
        # vec_2 = obj_pos - self.get_body_com("goal")

        # reward_near = -np.sum(np.abs(vec_1))
        # reward_dist = -np.sum(np.abs(vec_2))
        # reward_ctrl = -np.square(a).sum()
        # reward = 1.25 * reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near
        reward = -1 * self.cost_np(None, a, None)
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, {}

    def step(self, a):
        return self._step(a)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        qpos = self.init_qpos

        self.goal_pos = np.asarray([0, 0])
        self.cylinder_pos = np.array([-0.25, 0.15]) + np.random.normal(0, 0.025, [2])

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                high=0.005, size=self.model.nv)
        qvel[-4:] = 0
        init_state = np.vstack([qpos, qvel, np.zeros(shape=(17,1))])
        self.reset_mujoco(init_state)
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        self.ac_goal_pos = self.get_body_com("goal")

        return self._get_obs()

                # self.reset_mujoco(get_original_representation(init_state))
        # self.model.forward()
        # # qpos = self.init_qpos

        # self.goal_pos = np.asarray([0, 0])
        # self.cylinder_pos = np.array([-0.25, 0.15]) + np.random.normal(0, 0.025, [2])

        # qpos[-4:-2] = self.cylinder_pos
        # qpos[-2:] = self.goal_pos
        # qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
        #         high=0.005, size=self.model.nv)
        # qvel[-4:] = 0
        # self.set_state(qpos, qvel)
        # self.ac_goal_pos = self.get_body_com("goal")



    def reset(self):
        return self.reset_model()

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            self.get_body_com("object"),
        ])


    def get_obs(self):
        return self._get_obs()
    
    def get_current_obs(self):
        return self._get_obs()

    def cost_np(self, x, u, x_next, ctrl_cost_coeff=.1):
        obj_pos = self.get_body_com("object")
        vec_1 = obj_pos - self.get_body_com("tips_arm")
        vec_2 = obj_pos - self.get_body_com("goal")
        reward_near = -np.sum(np.abs(vec_1))
        reward_dist = -np.sum(np.abs(vec_2))
        reward_ctrl = -np.square(u).sum()
        reward = 1.25 * reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near

        return reward

    # def cost_tf(self, x, u, x_next, ctrl_cost_coeff=.1):

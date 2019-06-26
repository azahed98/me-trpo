from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
import tensorflow as tf
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
        qpos = self.init_qpos.copy().flatten()

        self.goal_pos = np.asarray([0, 0])
        self.cylinder_pos = np.array([-0.25, 0.15]) + np.random.normal(0, 0.025, [2])

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel.copy().flatten() + np.random.uniform(low=-0.005,
                high=0.005, size=self.model.nv)
        qvel[-4:] = 0
        init_state = np.vstack([qpos[:,None], qvel[:,None], np.zeros(shape=(17,1))])
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

    def get_EE_pos(self, states):
        theta1, theta2, theta3, theta4, theta5, theta6, theta7 = \
            states[:, :1], states[:, 1:2], states[:, 2:3], states[:, 3:4], states[:, 4:5], states[:, 5:6], states[:, 6:]

        rot_axis = np.concatenate([np.cos(theta2) * np.cos(theta1), np.cos(theta2) * np.sin(theta1), -np.sin(theta2)],
                                  axis=1)
        rot_perp_axis = np.concatenate([-np.sin(theta1), np.cos(theta1), np.zeros(theta1.shape)], axis=1)
        cur_end = np.concatenate([
            0.1 * np.cos(theta1) + 0.4 * np.cos(theta1) * np.cos(theta2),
            0.1 * np.sin(theta1) + 0.4 * np.sin(theta1) * np.cos(theta2) - 0.188,
            -0.4 * np.sin(theta2)
        ], axis=1)

        for length, hinge, roll in [(0.321, theta4, theta3), (0.16828, theta6, theta5)]:
            perp_all_axis = np.cross(rot_axis, rot_perp_axis)
            x = np.cos(hinge) * rot_axis
            y = np.sin(hinge) * np.sin(roll) * rot_perp_axis
            z = -np.sin(hinge) * np.cos(roll) * perp_all_axis
            new_rot_axis = x + y + z
            new_rot_perp_axis = np.cross(new_rot_axis, rot_axis)
            new_rot_perp_axis[np.linalg.norm(new_rot_perp_axis, axis=1) < 1e-20] = \
                rot_perp_axis[np.linalg.norm(new_rot_perp_axis, axis=1) < 1e-20]
            new_rot_perp_axis /= np.linalg.norm(new_rot_perp_axis, axis=1, keepdims=True)
            rot_axis, rot_perp_axis, cur_end = new_rot_axis, new_rot_perp_axis, cur_end + length * new_rot_axis

        return cur_end

    def get_EE_pos_tf(self, states):
        # raise NotImplementedError
        theta1, theta2, theta3, theta4, theta5, theta6, theta7 = \
            states[:, :1], states[:, 1:2], states[:, 2:3], states[:, 3:4], states[:, 4:5], states[:, 5:6], states[:, 6:]

        rot_axis = tf.concat([tf.cos(theta2) * tf.cos(theta1), tf.cos(theta2) * tf.sin(theta1), -tf.sin(theta2)],
                                  axis=1)
        rot_perp_axis = tf.concat([-tf.sin(theta1), tf.cos(theta1), tf.zeros_like(theta1)], axis=1)
        cur_end = tf.concat([
            0.1 * tf.cos(theta1) + 0.4 * tf.cos(theta1) * tf.cos(theta2),
            0.1 * tf.sin(theta1) + 0.4 * tf.sin(theta1) * tf.cos(theta2) - 0.188,
            -0.4 * tf.sin(theta2)
        ], axis=1)

        for length, hinge, roll in [(0.321, theta4, theta3), (0.16828, theta6, theta5)]:
            perp_all_axis = tf.cross(rot_axis, rot_perp_axis)
            x = tf.cos(hinge) * rot_axis
            y = tf.sin(hinge) * tf.sin(roll) * rot_perp_axis
            z = -tf.sin(hinge) * tf.cos(roll) * perp_all_axis
            new_rot_axis = x + y + z
            new_rot_perp_axis = tf.cross(new_rot_axis, rot_axis)
            new_rot_perp_axis_norm = tf.linalg.norm(new_rot_perp_axis, axis=1, keepdims=True)
            condition = tf.less(new_rot_perp_axis_norm, 1e-20)
            new_rot_perp_axis_norm = tf.where(condition, tf.linalg.norm(rot_perp_axis,axis=1, keepdims=True), new_rot_perp_axis_norm )
            # new_rot_perp_axis[tf.linalg.norm(new_rot_perp_axis, axis=1) < 1e-30] = \
            #     rot_perp_axis[tf.linalg.norm(new_rot_perp_axis, axis=1) < 1e-30]
            new_rot_perp_axis /= new_rot_perp_axis_norm#tf.linalg.norm(new_rot_perp_axis, axis=1, keepdims=True)
            rot_axis, rot_perp_axis, cur_end = new_rot_axis, new_rot_perp_axis, cur_end + length * new_rot_axis

        return cur_end

    def cost_np(self, x, u, x_next, ctrl_cost_coeff=.1):
        # obj_pos = self.get_body_com("object")
        # vec_1 = obj_pos - self.get_EE_pos(x_next)
        # vec_2 = obj_pos - self.get_body_com("goal")
        obj_pos = x_next[:, -3:]
        ee_pos = x_next[:, -6:-3]
        goal_pos = self.get_body_com("goal").reshape((1,3))
        vec_1 = obj_pos - ee_pos
        vec_2 = obj_pos - goal_pos
        reward_near = -np.sum(np.abs(vec_1))
        reward_dist = -np.sum(np.abs(vec_2))
        reward_ctrl = -np.square(u).sum()
        reward = 1.25 * reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near
        print("\nCOSTS:", self.old_cost_np(x, u, x_next), reward)

    def old_cost_np(self, x, u, x_next, ctrl_cost_coeff=.1):
        obj_pos = self.get_body_com("object")
        vec_1 = obj_pos - self.get_body_com("tips_arm")
        vec_2 = obj_pos - self.get_body_com("goal")
        reward_near = -np.sum(np.abs(vec_1))
        reward_dist = -np.sum(np.abs(vec_2))
        reward_ctrl = -np.square(u).sum()
        reward = 1.25 * reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near

        return reward

    def cost_tf(self, x, u, x_next, ctrl_cost_coeff=.1):
        obj_pos = x_next[:, -3:]
        ee_pos = x_next[:, -6:-3]
        goal_pos = tf.constant(self.get_body_com("goal").reshape((1,3)), dtype=tf.float32)
        vec_1 = obj_pos - ee_pos
        vec_2 = obj_pos - goal_pos
        # vec_1 = obj_pos - self.get_EE_pos_tf(x_next)
        # vec_2 = obj_pos - self.get_body_com("goal")
        reward_near = -tf.reduce_sum(np.abs(vec_1), axis=1)
        reward_dist = -tf.reduce_sum(np.abs(vec_2), axis=1)
        reward_ctrl = -tf.reduce_sum(tf.square(u), axis=1)
        reward = 1.25 * reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near
        return reward

    def cost_np_vec(self, x, u, x_next, ctrl_cost_coeff=.1):
        cost = np.array([self.cost_np(None, u[i], x_next[i], ctrl_cost_coeff) for i in range(x_next.shape[0])])
        return cost
    # def cost_tf(self, x, u, x_next, ctrl_cost_coeff=.1):

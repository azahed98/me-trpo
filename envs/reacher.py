from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
import tensorflow as tf
from gym import utils
from rllab.envs.mujoco import mujoco_env

class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(
            self,
            ctrl_cost_coeff=1e-2,
            *args, **kwargs):
        self.viewer = None
        utils.EzPickle.__init__(self)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.goal = np.zeros(3)
        # mujoco_env.MujocoEnv.__init__(self, os.path.join(dir_path, 'assets/reacher3d.xml'), 2)
        # self.sim = None
        super(ReacherEnv, self).__init__(*args, **kwargs, file_path='/home/azahed/me-trpo/vendor/mujoco_models/reacher3d.xml')

    # @property
    # def n_goals(self):
    #     '''
    #     :return: goal dimensions
    #     '''
    #     return 3
    @property
    def n_states(self):
        '''
        :return: state dimensions
        '''
        return 17

    
    def _step(self, a):
        # self.do_simulation(a, self.frame_skip)
        self.forward_dynamics(a)
        ob = self._get_obs()
        # print(ob.shape)
        # reward = np.sum(np.square(self.get_EE_pos(ob[None]) - self.goal))
        # reward += 0.1 * np.square(a).sum()
        reward = self.cost_np(None, a, ob[None])
        # print("OBS REWARD")
        # print(reward)
        done = False
        return ob, reward * -1 , done, dict(reward_dist=0, reward_ctrl=0)

    def step(self, a):
        return self._step(a)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = 2.5
        self.viewer.cam.elevation = -30
        self.viewer.cam.azimuth = 270

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,1) and qvel.shape == (self.model.nv,1)
        # oldqpos = np.copy(self.model.data.qpos)
        # oldqvel = np.copy(self.model.data.qvel)
        if self.model.na == 0:
            act = None
        else:
            act = np.copy(self.model.data.act)
     
        self.model.data.qpos[:] = np.copy(qpos)
        self.model.data.qvel[:] = np.copy(qvel)
        if self.model.na != 0:
            self.model.data.act[:] = np.copy(act)

        self.model.forward()

    def reset_model(self, init_pos=None):
        qpos, qvel = np.copy(self.init_qpos), np.copy(self.init_qvel)
        qpos[-3:] += np.random.normal(loc=0, scale=0.1, size=[3,1])
        qvel[-3:] = 0
        self.goal = qpos[-3:]
        init_state = np.vstack([qpos, qvel, np.zeros(shape=(17,1))])
        self.reset_mujoco(init_state)
        # self.set_state(qpos, qvel)
        # self.reset_mujoco(get_original_representation(init_state))
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        return self.get_current_obs()
        # self.set_state(qpos, qvel)

    def reset(self, init_pos=None):
        return self.reset_model(init_pos)

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat[:-3]
        ])

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
        # return self.cost_np_vec(x, u.reshape(1,-1), x_next.reshape(1,-1), ctrl_cost_coeff)
        # assert len(x_next.shape) <= 2
        if len(x_next.shape) == 2:
            # assert x_next.shape[0] <= 1, str(x_next.shape)
            return np.mean(self.cost_np_vec(x, u, x_next, ctrl_cost_coeff))
        cost = np.sum(np.square(self.get_EE_pos(x_next.reshape(1,-1)) - self.goal.T))
        cost += ctrl_cost_coeff * np.square(u).sum()
        if np.isnan(cost).any():
            print(x, u, x_next, ctrl_cost_coeff)
            input("NAN DETECTED!!!!\n")
        return cost
        # return np.mean(np.linalg.norm(x[:, -2:]-get_fingertips(x), axis=1) +\
        #                ctrl_cost_coeff*0.5*np.sum(np.square(u), axis=1))

    def cost_tf(self, x, u, x_next, ctrl_cost_coeff=.1):
        cost = tf.reduce_sum(tf.square(self.get_EE_pos_tf(x_next) - self.goal.T), axis=1)
        cost += ctrl_cost_coeff * tf.reduce_sum(tf.square(u), axis=1)
        return cost
        # return tf.reduce_mean(tf.norm(x[:, -2:]-get_fingertips_tf(x), axis=1) +\
        #                ctrl_cost_coeff*0.5*tf.reduce_sum(tf.square(u), axis=1))

    def cost_np_vec(self, x, u, x_next, ctrl_cost_coeff=.1):
        # cost = np.sum(np.square(self.get_EE_pos(x_next) -  self.goal.T), axis=1)
        # # cost = np.square(np.apply_along_axis(lambda a: self.get_EE_pos(a.reshape(1, -1)), 1, x_next) -  self.goal.T)
        # # cost = np.sum(np.sum(cost, axis=1), axis=1)
        # if len(u.shape) == 1:
        #     assert x_next.shape[0] == 1
        #     cost += ctrl_cost_coeff * np.sum(np.square(u))
        # else:
        #     cost += ctrl_cost_coeff * np.sum(np.square(u), axis=1)
        # if np.isnan(cost).any():
        #     print(u, "\n", x_next, "\n", 
        #          np.apply_along_axis(lambda a: self.get_EE_pos(a.reshape(1, -1)), 1, x_next),
        #          "\n", ctrl_cost_coeff)
        #     input("NAN DETECTED!!!!\n")
        # print(u, "\n", x_next, "\n", 
        #          np.apply_along_axis(lambda a: self.get_EE_pos(a.reshape(1, -1)), 1, x_next),
        #          "\n", ctrl_cost_coeff)
        # if len(x_next.shape) == 1:
        #     return cost_np(x, u, x_next, ctrl_cost_coeff)
        # if len(u.shape) == 1:
        #     return cost_np(None, u, x_next[0], ctrl_cost_coeff)
        cost = np.array([self.cost_np(None, u[i], x_next[i], ctrl_cost_coeff) for i in range(x_next.shape[0])])
        return cost
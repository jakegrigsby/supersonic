from copy import deepcopy
import os

import numpy as np
import tensorflow as tf
from mpi4py import MPI

from supersonic import environment, utils, models, logger

class PPOAgent:
    """
    Basic version of Proximal Policy Optimization (Clip) with exploration by Random Network Distillation.
    """
    def __init__(self, env_id, exp_lr=.001, ppo_lr=.0001, vis_model='NatureVision', policy_model='NaturePolicy', val_model='VanillaValue',
                    exp_target_model='NatureVision', exp_train_model='NatureVision', exp_epochs=4, gamma_i=.99, gamma_e=.999, log_dir=None,
                        rollout_length=128, ppo_epochs=4, e_rew_coeff=2., i_rew_coeff=1., vf_coeff=.25, exp_train_prop=.25, lam=.95, exp_batch_size=32,
                    ppo_batch_size=32, ppo_clip_value=0.1, checkpoint_interval=1000, minkl=None, entropy_coeff=.001, random_actions=0, max_grad_norm=0.):

        tf.enable_eager_execution()
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.env = environment.auto_env(env_id)
        utils.random_actions(self.env, random_actions)

        try:
            self.env_is_sonic = self.env.SONIC
        except:
            self.env_is_sonic = False

        self.most_recent_step = self.env.reset(), 0, False, {}
        self.nb_actions = self.env.action_space.n

        self.exp_lr = exp_lr
        self.ppo_lr = ppo_lr
        self.exp_optimizer = tf.train.AdamOptimizer(learning_rate=self.exp_lr)
        self.ppo_optimizer = tf.train.AdamOptimizer(learning_rate=self.ppo_lr)
        self.max_grad_norm = max_grad_norm
        self.minkl = minkl
        self.stop = False
        self.checkpoint_interval = checkpoint_interval

        self.vis_model = models.get_model(vis_model)()
        self.policy_model = models.get_model(policy_model)(self.nb_actions)
        self.val_model_e = models.get_model(val_model)()
        self.val_model_i = models.get_model(val_model)()
        self.exp_target_model = models.get_model(exp_target_model)()
        self.exp_train_model = models.get_model(exp_train_model)()

        self.exp_epochs = exp_epochs
        self.exp_train_prop = exp_train_prop
        self.exp_batch_size = exp_batch_size
        self.ppo_epochs = ppo_epochs
        self.ppo_batch_size = ppo_batch_size
        self.clip_value = ppo_clip_value
        self.vf_coeff = vf_coeff
        self.entropy_coeff = entropy_coeff

        self.gamma_i = gamma_i
        self.gamma_e = gamma_e
        self.lam = lam
        self.e_rew_coeff = e_rew_coeff
        self.i_rew_coeff = i_rew_coeff
        self.rollout_length = rollout_length

        if self.rank == 0:
            self.log_dir = os.path.join('logs',log_dir) if log_dir else 'logs/'
            self.logger = logger.Logger(self.log_dir)
            self._log_episode_num = 0
            self._reset_stats()
    
    def _gather_array(self, array, dtype=np.float32):
        """
        Move a numpy array from a worker node to rank 0
        """
        recv_buff = None
        if self.rank == 0:
            recv_buff = np.empty([self.size] + list(array.shape), dtype=dtype)
        self.comm.Gather(array, recv_buff, root=0)
        return recv_buff
    
    def _reshape_trajectory(self, trajectory):
        """
        Input: trajectory assembled by MPI Gather op
        Output: trajectory reshaped to be normal trajectory dimensions, but with the data from every worker.
        """
        trajectory.states = np.reshape(trajectory.states, (-1,) + trajectory.states.shape[2:])
        trajectory.rews_e = np.expand_dims(trajectory.rews_e.flatten(), -1)
        trajectory.rews_i = np.expand_dims(trajectory.rews_i.flatten(), -1)
        trajectory.gaes = np.squeeze(trajectory.gaes.flatten())
        trajectory.old_act_probs = trajectory.old_act_probs.flatten()
        trajectory.exp_targets = np.reshape(trajectory.exp_targets, (-1, trajectory.exp_targets.shape[-1]))
        trajectory.actions = trajectory.actions.flatten()
        return trajectory

    def train(self, rollouts, render=0):
        past_trajectory = None
        if self.rank == 0: progbar = tf.keras.utils.Progbar(rollouts)
        for rollout in range(rollouts):
            trajectory = self._rollout(self.rollout_length, past_trajectory, render=True if self.rank <= render else False)
            self.comm.barrier()
            #consolidate each workers' trajectories into one dataset we can train on
            super_trajectory = utils.Trajectory(self.rollout_length)
            super_trajectory.rews_e = self._gather_array(trajectory.rews_e)
            super_trajectory.rews_i = self._gather_array(trajectory.rews_i)
            super_trajectory.gaes = self._gather_array(trajectory.gaes)
            super_trajectory.old_act_probs = self._gather_array(trajectory.old_act_probs)
            super_trajectory.exp_targets = self._gather_array(trajectory.exp_targets)
            super_trajectory.actions = self._gather_array(trajectory.actions, dtype=np.int64)
            super_trajectory.states = self._gather_array(trajectory.states)
            self.comm.barrier()
            if self.rank == 0:
                super_trajectory = self._reshape_trajectory(super_trajectory)
                self._update_models(super_trajectory)
            #broadcast new weights to workers
            self.comm.barrier()
            self.weights = self.comm.bcast(self.weights, root=0)
            if self.stop: exit() #if early stopping is activated
            past_trajectory = deepcopy(trajectory)
            if self.rank == 0:
                progbar.update(rollout+1)
                if rollout % self.checkpoint_interval == 0:
                    self._checkpoint(rollout)
            self.comm.barrier()
        self.save_weights('final')

    def test(self, episodes, max_ep_steps=4500, render=False, stochastic=True):
        try:
            self.env.max_steps = max_ep_steps
        except:
            pass
        cum_rew = 0
        for ep in range(episodes):
            step = 0
            obs, rew, done, info = self.env.reset(), 0, False, {}
            while step < max_ep_steps and not done:
                action = self._choose_action(obs, stochastic=stochastic)
                obs, rew, done, info = self.env.step(action)
                if render: self.env.render()
                cum_rew += rew
                step += 1
        return cum_rew
    
    def _rollout(self, steps, past_trajectory=None, render=False):
        """
        Step through the environment using current policy. Calculate all the metrics needed by update_models() and save it to a util.Trajectory object
        """
        if not self.most_recent_step[2]: #if not done
            obs, e_rew, done, info = self.most_recent_step
        else:
            obs, e_rew, done, info = self.env.reset(), 0, False, {}
        step = 0
        trajectory = utils.Trajectory(self.rollout_length, past_trajectory)
        while step < steps:
            if done: 
                obs, e_rew, done, info = self.env.reset(), 0, False, {} #trajectories roll through the end of episodes
            action_prob, action, val_e, val_i = self._choose_action_get_value(obs)
            val_e, val_i = (val_e, val_i) if not done else (0, 0)
            obs2, e_rew, done, info = self.env.step(action)
            if render: self.env.render()
            i_rew, exp_target = self._calc_intrinsic_reward(obs)
            trajectory.add(obs, e_rew, i_rew, exp_target, (action_prob, action), val_e, val_i)
            if self.env_is_sonic and self.rank == 0:
                self._update_ep_stats(action, e_rew, i_rew, done, info)
            obs = obs2
            step += 1
        _, _, last_val_e, last_val_i = self._choose_action_get_value(obs) if not done else (0, 0, 0, 0)
        self.most_recent_step = (obs, e_rew, done, info)
        trajectory.end_trajectory(self.e_rew_coeff, self.i_rew_coeff, self.gamma_i, self.gamma_e, self.lam, last_val_i, last_val_e)
        return trajectory

    def _reset_stats(self):
        self._log_action_count = [0 for i in range(self.nb_actions)]
        self._log_cum_rew_e, self._log_cum_rew_i = 0, 0
        self._log_furthest_point = (0,0)
        self._log_death_coords = []
        self._log_current_lives = 3
        self._log_episode_num += 1
        self._log_training_steps = 0

    def _update_ep_stats(self, action, e_rew, i_rew, done, info):
        self._log_action_count[action] += 1
        self._log_cum_rew_e += e_rew
        self._log_cum_rew_i += i_rew
        self._log_training_steps += 1
        if info['lives'] < self._log_current_lives:
            self._log_current_lives -= 1
            current_pos = (info['screen_x'], info['screen_y'])
            self._log_death_coords.append(current_pos)
            self._log_furthest_point = max(self._log_furthest_point, current_pos)
        if done:
            if "FORCED EXIT" in info:
                self._log_death_coords = (info['screen_x'], info['screen_y'])
                self._log_furthest_point = self._log_death_coords
            episode_dict = {'episode_num':self._log_episode_num,
                            'death_coords':self._log_death_coords,
                            'training_steps':self._log_training_steps,
                            'max_x':self._log_furthest_point[0],
                            'score':info['score'],
                            'external_reward':self._log_cum_rew_e,
                            'internal_reward':self._log_cum_rew_i,
                            'action_count':self._log_action_count,}
            episode_log = logger.EpisodeLog(episode_dict)
            self.logger.log_episode(episode_log)
            self._reset_stats()

    def _choose_action(self, obs, stochastic=True):
        """
        Choose an action based on the current observation. Saves computation by not running the value net,
        which makes it a good choice for testing the agent.
        """
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        features = self.vis_model(obs)
        action_probs = np.squeeze(self.policy_model(features)) + 1e-8
        if stochastic:
            action = np.random.choice(np.arange(self.nb_actions), p=action_probs)
        else:
            action = np.argmax(action_probs)
        return action

    def _choose_action_get_value(self, obs):
        """
        Run the entire network and return the action probability distribution (not just the chosen action) as well as the values
        from the value net. Used during training -  when more information is needed.
        """
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        features = self.vis_model(obs)
        action_probs = np.squeeze(np.transpose(self.policy_model(features)), axis=-1) + 1e-8
        action_idx = np.random.choice(np.arange(self.nb_actions), p=action_probs)
        action_prob = action_probs[action_idx]
        val_e = tf.squeeze(self.val_model_e(features))
        val_i = tf.squeeze(self.val_model_i(features))
        return tf.log(action_prob), action_idx, val_e, val_i

    def _calc_intrinsic_reward(self, state):
        """
        reward as described in Random Network Distillation
        """
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        target = self.exp_target_model(state)
        pred = self.exp_train_model(state)
        rew = np.mean(np.square(target - pred))
        return rew, target #save targets to trajectory to avoid another call to the exp_target_model network during update_models()

    def _checkpoint(self, rollout_num):
        save_path = 'weights/{}/checkpoint_{}'.format(os.path.basename(self.log_dir), rollout_num)
        if not os.path.exists(save_path): os.makedirs(save_path)
        self.save_weights(save_path)

    def _update_models(self, trajectory):
        """
        Update the vision, policy, value, and exploration (RND) networks based on a utils.Trajectory object generated by a rollout.
        """
        #create new training set
        exp_train_samples = int(self.rollout_length * self.exp_train_prop)
        idxs = np.random.choice(trajectory.states.shape[0], exp_train_samples, replace=False)
        dataset = tf.data.Dataset.from_tensor_slices((np.take(trajectory.states, idxs, axis=0), np.take(trajectory.exp_targets, idxs, axis=0)))
        dataset = dataset.shuffle(exp_train_samples+1).repeat(self.exp_epochs).batch(self.exp_batch_size)

        #update exploration net
        for (batch, (state, target)) in enumerate(dataset):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.square(target - self.exp_train_model(state)))
            grads = tape.gradient(loss, self.exp_train_model.variables)
            self.exp_optimizer.apply_gradients(zip(grads, self.exp_train_model.variables))

        #create new training set
        dataset = tf.data.Dataset.from_tensor_slices((trajectory.states, trajectory.rews_e, trajectory.rews_i, trajectory.old_act_probs, trajectory.actions, trajectory.gaes))
        dataset = dataset.shuffle(trajectory.states.shape[0]+1).repeat(self.ppo_epochs).batch(self.ppo_batch_size)
        #update policy and value nets
        for (batch, (state, e_rew, i_rew, old_act_prob, action, gae)) in enumerate(dataset):
            with tf.GradientTape() as tape:
                row_idxs = tf.range(action.shape[0], dtype=tf.int64)
                action = tf.stack([row_idxs, tf.squeeze(action)], axis=1)
                features = self.vis_model(state)
                new_act_probs = self.policy_model(features)
                new_act_prob = tf.log(tf.gather_nd(new_act_probs, action))
                old_act_prob = tf.squeeze(old_act_prob)

                val_e = self.val_model_e(features)
                val_e_loss = tf.reduce_mean(tf.square(e_rew - val_e))
                val_i = self.val_model_i(features)
                val_i_loss = tf.reduce_mean(tf.square(i_rew - val_i))
                v_loss = val_e_loss + val_i_loss

                ratio = tf.exp(new_act_prob - old_act_prob)
                min_gae = tf.where(gae >= 0, (1+self.clip_value)*gae, (1-self.clip_value)*gae)
                p_loss = -tf.reduce_mean(tf.minimum(ratio * gae, min_gae))
                entropy = tf.reduce_mean(-new_act_prob)

                loss = p_loss + self.vf_coeff*v_loss - self.entropy_coeff*entropy

            #backprop and apply grads
            variables = self.vis_model.variables + self.policy_model.variables + self.val_model_e.variables + self.val_model_i.variables
            grads = tape.gradient(loss, variables)
            if self.max_grad_norm:
                grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
            self.ppo_optimizer.apply_gradients(zip(grads, variables))

            approxkl = tf.reduce_mean(old_act_prob - new_act_prob)
            if self.minkl and approxkl < self.minkl:
                self._checkpoint('earlyStopped')
                self.stop = True

    def save_weights(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.vis_model.save_weights(os.path.join(path, 'vis_model'))
        self.policy_model.save_weights(os.path.join(path, 'pol_model'))
        self.val_model_e.save_weights(os.path.join(path, 'val_model_e'))
        self.val_model_i.save_weights(os.path.join(path, 'val_model_i'))
        self.exp_train_model.save_weights(os.path.join(path, 'exp_train_model'))
        self.exp_target_model.save_weights(os.path.join(path, 'exp_target_model'))

    def load_weights(self, path):
        self.vis_model.load_weights(os.path.join(path, 'vis_model'))
        self.policy_model.load_weights(os.path.join(path, 'pol_model'))
        self.val_model_e.load_weights(os.path.join(path, 'val_model_e'))
        self.val_model_i.load_weights(os.path.join(path, 'val_model_i'))
        self.exp_train_model.load_weights(os.path.join(path, 'exp_train_model'))
        self.exp_target_model.load_weights(os.path.join(path, 'exp_target_model'))

    @property
    def weights(self):
        return [self.vis_model.get_weights(),
                self.policy_model.get_weights(),
                self.val_model_e.get_weights(),
                self.val_model_i.get_weights(),
                self.exp_target_model.get_weights(),
                self.exp_train_model.get_weights()]

    @weights.setter
    def weights(self, new_weights):
        self.vis_model.set_weights(new_weights[0])
        self.policy_model.set_weights(new_weights[1])
        self.val_model_e.set_weights(new_weights[2])
        self.val_model_i.set_weights(new_weights[3])
        self.exp_target_model.set_weights(new_weights[4])
        self.exp_train_model.set_weights(new_weights[5])
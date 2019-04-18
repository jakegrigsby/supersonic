import os

import tensorflow as tf
import numpy as np

from supersonic import environment, utils, models, logger


class DQNAgent:

    def __init__(self, env_id, memory_size=1000000, q_network='NoisyQNetwork', exp_target_network='NatureVision', exp_train_network='NatureVision',
                 dqn_lr=.0001, exp_lr=.001, gamma=.99, batch_size=32, nb_steps_warmup=1000, random_start_steps=10, train_interval=4, target_model_update=10000,
                 delta_clip=1., memory_start_beta=1., memory_end_beta=1., memory_alpha=.4,  memory_steps_annealed=1, rew_coeff_e=2., rew_coeff_i=1.,
                 exp_net_opt_steps=4, dqn_net_opt_steps=4, n_step=4, log_dir=None):

        tf.enable_eager_execution()

        self.env = environment.auto_env(env_id)
        self.nb_actions = self.env.action_space.n
        self.random_start_steps = random_start_steps if random_start_steps > 0 else 1

        self.dqn_lr = dqn_lr
        self.exp_lr = exp_lr
        self.model = models.get_model(q_network)(self.nb_actions)
        self.target_model = models.get_model(q_network)(self.nb_actions)
        self.exp_train_model = models.get_model(exp_train_network)()
        self.exp_target_model = models.get_model(exp_target_network)()
        self.exp_net_opt_steps = exp_net_opt_steps
        self.batch_size = batch_size
        self.exp_optimizer = tf.train.AdamOptimizer(learning_rate=self.exp_lr)
        self.dqn_optimizer = tf.train.AdamOptimizer(learning_rate=self.dqn_lr)

        self.memory = utils.PrioritizedMemory(memory_size, memory_alpha, memory_start_beta, memory_end_beta, memory_steps_annealed)

        self.train_interval = train_interval
        self.target_model_update = target_model_update
        self.delta_clip = delta_clip
        self.n_step = n_step

        self.rew_coeff_e = rew_coeff_e
        self.rew_coeff_i = rew_coeff_i

        self.log_dir = os.path.join('logs', log_dir) if log_dir else 'logs/'
        self.logger = logger.Logger(self.log_dir)
        self._log_episode_num = 0
        self._reset_stats()

        self._warmup(nb_steps_warmup)

    def train(self, steps, device='/cpu:0', render=False):
        self.train_step = 0
        obs, rew, done, info = utils.random_actions(self.env, 1)
        while self.train_step < steps:
            if done:
                obs = self.env.reset()
                obs, rew, done, info = utils.random_actions(
                    self.random_start_steps)
            action = self._choose_action(obs)
            rew_i, exp_target = self._calc_intrinsic_reward(obs)
            obs, rew_e, done, info = self.env.step(action)
            if render: self.env.render()
            rew = self.rew_coeff_e*rew_e + self.rew_coeff_i*rew_i
            self.memory.append(obs, action, rew, done, exp_target)
            self._update_ep_stats(action, rew_e, rew_i, done, info)
            self._update_models(device=device)
            self.train_step += 1

    def _reset_stats(self):
        self._log_action_count = [0 for i in range(self.nb_actions)]
        self._log_cum_rew_e, self._log_cum_rew_i = 0, 0
        self._log_furthest_point = (0, 0)
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

    def _warmup(self, steps):
        for _ in range(steps):
            action = self.env.action_space.sample()
            obs, rew_e, done, info = self.env.step(action)
            import pdb; pdb.set_trace()
            rew_i, exp_target = self._calc_intrinsic_reward(obs)
            rew = self.rew_coeff_e*rew_e + self.rew_coeff_i*rew_i
            self.memory.append(obs, action, rew, done, exp_target)
            if done:
                self.env.reset()

    def _choose_action(self, obs):
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        q_values = self.model(obs)
        action_idx = np.argmax(q_values)
        return action_idx

    def _calc_intrinsic_reward(self, state):
        """
        reward as described in Random Network Distillation
        """
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        target = self.exp_target_model(state)
        pred = self.exp_train_model(state)
        rew = np.mean(np.square(target - pred))
        return rew, target  # save targets to trajectory to avoid another call to the exp_target_model network during update_models()

    def _calc_double_q_values(self, state1_batch):
        # According to the paper "Deep Reinforcement Learning with Double Q-learning"
        # (van Hasselt et al., 2015), in Double DQN, the online network predicts the actions
        # while the target network is used to estimate the Q value.
        q_values = self.model(state1_batch)
        assert q_values.shape == (self.batch_size, self.nb_actions)
        actions = np.argmax(q_values, axis=1)
        assert actions.shape == (self.batch_size,)

        # Now, estimate Q values using the target network but select the values with the
        # highest Q value wrt to the online model (as computed above).
        target_q_values = self.target_model(state1_batch)
        assert target_q_values.shape == (self.batch_size, self.nb_actions)
        q_batch = target_q_values[range(self.batch_size), actions]

        return q_batch

    def _update_models(self, device):
        current_beta = self.memory.calculate_beta(self.train_step)
        experiences = self.memory.sample(self.batch_size, current_beta)

        state0_batch, reward_batch, action_batch, terminal1_batch, state1_batch = [], [], [], [], []
        for e in experiences[:-3]:
            state0_batch.append(e.state0)
            state1_batch.append(e.state1)
            reward_batch.append(e.reward)
            action_batch.append(e.action)
            terminal1_batch.append(0. if e.terminal1 else 1.)
        importance_weights = experiences[-3]
        pr_idxs = experiences[-2]
        exp_targets = experiences[-1]
        terminal1_batch = np.array(terminal1_batch)
        reward_batch = np.array(reward_batch)

        state1_batch = np.array(state1_batch, dtype=np.float32)
        state0_batch = np.array(state0_batch, dtype=np.float32)
        q_batch = self._calc_double_q_values(state1_batch)
        q_batch_n = self._calc_double_q_values(state0_batch)

        #Multi-step loss targets
        targets_n = np.zeros((self.batch_size, self.nb_actions))
        masks = np.zeros((self.batch_size, self.nb_actions))
        dummy_targets_n = np.zeros((self.batch_size,))
        discounted_reward_batch_n = (self.gamma**self.n_step) * q_batch_n
        discounted_reward_batch_n *= terminal_batch_n
        assert discounted_reward_batch_n.shape == reward_batch_n.shape
        Rs_n = reward_batch_n + discounted_reward_batch_n
        for idx, (target, mask, R, action) in enumerate(zip(targets_n, masks, Rs_n, action_batch)):
            target[action] = R  # update action with estimated accumulated reward
            dummy_targets_n[idx] = R
            mask[action] = 1.  # enable loss for this specific action
        targets_n = np.array(targets_n).astype('float32')

        #Single-step loss targets
        targets = np.zeros((self.batch_size, self.nb_actions))
        dummy_targets = np.zeros((self.batch_size,))
        discounted_reward_batch = (self.gamma) * q_batch
        discounted_reward_batch *= terminal1_batch
        assert discounted_reward_batch.shape == reward_batch.shape
        Rs = reward_batch + discounted_reward_batch
        for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, action_batch)):
            target[action] = R  # update action with estimated accumulated reward
            dummy_targets[idx] = R
        targets = np.array(targets).astype('float32')
        masks = np.array(masks).astype('float32')


        importance_weights = np.array(importance_weights)
        importance_weights = np.vstack([importance_weights]*self.nb_actions)
        importance_weights = np.reshape(importance_weights, (self.batch_size, self.nb_actions))

        # update dqn network
        dataset = tf.data.Dataset.from_tensor_slices((targets, state0_batch, importance_weights, masks))
        for batch, (target, state, importance_weight, mask) in enumerate(dataset.take(self.dqn_net_opt_steps)):
            with tf.GradientTape() as tape:
                loss = tf.keras.backend.sum(tf.losses.huber_loss(target, self.model(state), delta=self.delta_clip) * importance_weight, axis=-1)
            grads = tape.gradient(loss, self.model.variables)
            self.dqn_optimizer.apply_gradients(zip(grads, self.model.variables))

        if self.train_step % self.target_model_update == 0: self._update_target_model()

        # update random network distillation network
        dataset = tf.data.Dataset.from_tensor_slices((state0_batch, pr_idxs, exp_targets))
        for batch, (state, pr_idx, target) in enumerate(dataset.take(self.exp_net_opt_steps)):
            with tf.GradientTape() as tape:
                loss = tf.losses.mean_squared_error(target, self.exp_train_model(state))
            grads = tape.gradient(loss, self.exp_train_model.variables)
            self.exp_optimizer.apply_gradients(zip(grads, self.exp_train_model.variables, global_step=tf.train.get_or_create_global_step()))
            # update prioritized memory
            self.memory.update_priorities(pr_idx, loss)

    def _update_target_model():
        """Copy weights from train model to target model"""
        self.target_model.set_weights(self.model.get_weights())

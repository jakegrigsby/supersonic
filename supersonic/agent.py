from copy import deepcopy
import os

import numpy as np
import tensorflow as tf

from supersonic import environment, utils, models, logger

def ppo_agent(env_id, hyp_dict, log_dir):
    """
    When tuning hyperparameters, it'll be easier to create and modify dictionaries, which can be passed between
    MPI processes more efficiently. This function takes a dictionary describing the agent's params, as well as a
    string representing the environment and a log dir, and creates a BaseAgent.
    """
    return BaseAgent(env_id, log_dir=log_dir, **hyp_dict)


class BaseAgent:
    """
    Basic version of Proximal Policy Optimization (Clip) with exploration by Random Network Distillation.
    """
    def __init__(self, env_id, exp_lr=.001, ppo_lr=.001, vis_model='NatureVision', policy_model='NaturePolicy', val_model='VanillaValue',
                    exp_target_model='NatureVision', exp_train_model='NatureVision', exp_net_opt_steps=None, gamma_i=.99, gamma_e=.999, log_dir=None,
                    rollout_length=128, ppo_net_opt_steps=8, e_rew_coeff=2., i_rew_coeff=1., exp_train_prop=.5, lam=.99, exp_batch_size=32,
                    ppo_batch_size=32, ppo_clip_value=0.2, update_mean_gae_until=10000, checkpoint_interval=10000):

        tf.enable_eager_execution()

        self.env = environment.auto_env(env_id)
        self.most_recent_step = self.env.reset(), 0, False, {}
        self.nb_actions = self.env.action_space.n

        self.exp_lr = exp_lr
        self.ppo_lr = ppo_lr
        self.checkpoint_interval = checkpoint_interval

        self.vis_model = models.get_model(vis_model)()
        self.policy_model = models.get_model(policy_model)(self.nb_actions)
        self.val_model_e = models.get_model(val_model)()
        self.val_model_i = models.get_model(val_model)()
        self.exp_target_model = models.get_model(exp_target_model)()
        self.exp_train_model = models.get_model(exp_train_model)()

        self.exp_net_opt_steps = exp_net_opt_steps if exp_net_opt_steps else max(int((rollout_length * exp_train_prop)/exp_batch_size), 1)
        self.exp_train_prop = exp_train_prop
        self.exp_batch_size = exp_batch_size
        self.ppo_net_opt_steps = ppo_net_opt_steps if ppo_net_opt_steps else max(int(rollout_length / ppo_batch_size), 1)
        self.ppo_batch_size = ppo_batch_size
        self.clip_value = ppo_clip_value

        self.gamma_i = gamma_i
        self.gamma_e = gamma_e
        self.lam = lam
        self.e_rew_coeff = e_rew_coeff
        self.i_rew_coeff = i_rew_coeff

        self.rollout_length = rollout_length

        self.update_mean_gae_until = update_mean_gae_until
        self._gae_count = 0
        self._gae_running_mean = np.zeros((rollout_length,1), dtype=np.float32)
        self._gae_m2 = np.zeros_like(self._gae_running_mean)

        self.log_dir = os.path.join('logs',log_dir) if log_dir else 'logs/'
        self.logger = logger.Logger(self.log_dir)
        self._log_episode_num = 0
        self._reset_stats()

    def train(self, rollouts, device='/cpu:0', render=False):
        past_trajectory = None
        for rollout in range(rollouts):
            print("Rollout #{}".format(rollout))
            trajectory = self._rollout(self.rollout_length, past_trajectory, render=render)
            self._update_models(trajectory, device)
            past_trajectory = deepcopy(trajectory)
            if self.checkpoint_interval and rollout % self.checkpoint_interval == 0:
                self._checkpoint(rollout)

    def test(self, episodes, max_ep_steps=4500, render=False, stochastic=True):
        cum_rew = 0
        for ep in range(episodes):
            step = 0
            obs, rew, done, info = self.env.reset(), 0, False, {}
            while step < max_ep_steps and not done:
                action = self._choose_action(obs, training=stochastic)
                obs, rew, done, info = self.env.step(action)
                if render: self.env.render()
                cum_rew += rew
                step += 1
        return cum_rew

    def _rollout(self, steps, past_trajectory=None, render=False):
        """
        Step through the environment using current policy. Calculate all the metrics needed by update_models() and save it to a util.Trajectory object
        """
        #pick up where the last rollout left off, unless we need to reset the environment
        if not self.most_recent_step[2]:
            obs, e_rew, done, info = self.most_recent_step
        else:
            obs, e_rew, done, info = self.env.reset(), 0, False, {}
        step = 0
        trajectory = utils.Trajectory(self.rollout_length, past_trajectory)
        while step < steps:
            if done: obs, e_rew, done, info = self.env.reset(), 0, False, {} #trajectories roll through the end of episodes
            action_prob, action, val_e, val_i = self._choose_action_get_value(obs)
            val_e, val_i = (val_e, val_i) if not done else (0, 0)
            obs2, e_rew, done, info = self.env.step(action)
            if render: self.env.render()
            i_rew, exp_target = self._calc_intrinsic_reward(obs)
            trajectory.add(obs, e_rew, i_rew, exp_target, (action_prob, action), val_e, val_i)
            self._update_ep_stats(action, e_rew, i_rew, done, info)
            step += 1
            obs = obs2
        _, _, last_val_e, last_val_i = self._choose_action_get_value(obs) if not done else (0, 0)
        self.most_recent_step = (obs, e_rew, done, info)
        trajectory.end_trajectory(self.gamma_i, self.gamma_e, self.lam, self.i_rew_coeff, self.e_rew_coeff, last_val_e, last_val_i)
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

    def _choose_action(self, obs, training=True):
        """
        Choose an action based on the current observation. Saves computation by not running the value net,
        which makes it a good choice for testing the agent.
        """
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        features = self.vis_model(obs)
        action_probs = np.squeeze(self.policy_model(features))
        if training:
            action = np.random.choice(np.arange(self.nb_actions), p=action_probs)
        else:
            action = np.argmax(action_probs)
        return action

    def _choose_action_get_value(self, obs, training=True):
        """
        Run the entire network and return the action probability distribution (not just the chosen action) as well as the values
        from the value net. Used during training -  when more information is needed.
        """
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        features = self.vis_model(obs)
        action_probs = np.squeeze(np.transpose(self.policy_model(features)), axis=-1)
        action_idx = np.random.choice(np.arange(self.nb_actions), p=action_probs)
        action_prob = action_probs[action_idx]
        val_e = tf.squeeze(self.val_model_e(features))
        val_i = tf.squeeze(self.val_model_i(features))
        return action_prob, action_idx, val_e, val_i

    def _calc_intrinsic_reward(self, state):
        """
        reward as described in Random Network Distillation
        """
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        target = self.exp_target_model(state)
        pred = self.exp_train_model(state)
        rew = np.square(np.subtract(target, pred)).mean()
        return rew, target #save targets to trajectory to avoid another call to the exp_target_model network during update_models()

    def _normalize_gaes(self, gaes):
        if self._gae_count < self.update_mean_gae_until:
            self._update_gae_normalization(gaes)
        var = self._gae_m2 / ((self._gae_count-1) + 1e-3)
        std = np.sqrt(var)
        gaes = (gaes - self._gae_running_mean) / (std + 1e-3)
        return gaes

    def _update_gae_normalization(self, gaes):
        self._gae_count += 1
        delta = gaes - self._gae_running_mean
        self._gae_running_mean += delta / self._gae_count
        delta2 = gaes - self._gae_running_mean
        self._gae_m2 += delta * delta2

    def _checkpoint(self, rollout_num):
        save_path = 'weights/{}/checkpoint_{}'.format(os.path.basename(self.log_dir), rollout_num)
        if not os.path.exists(save_path): os.makedirs(save_path)
        self.save_weights(save_path)

    def _update_models(self, trajectory, device='/cpu:0'):
        """
        Update the vision, policy, value, and exploration (RND) networks based on a utils.Trajectory object generated by a rollout.
        Should be able specify device so that multiple agents can be efficiently trained on the same (multi-gpu) machine.
        """
        with tf.device(device):
            #create new training set
            exp_train_samples = int(self.rollout_length * self.exp_train_prop)
            idxs = np.random.choice(trajectory.states.shape[0], exp_train_samples, replace=False)
            dataset = tf.data.Dataset.from_tensor_slices((np.take(trajectory.states, idxs, axis=0), np.take(trajectory.exp_targets, idxs, axis=0)))
            dataset = dataset.shuffle(100).batch(self.exp_batch_size)

            #update exploration net
            optimizer = tf.train.AdamOptimizer(learning_rate=self.exp_lr)
            for (batch, (state, target)) in enumerate(dataset.take(self.exp_net_opt_steps)):
                with tf.GradientTape() as tape:
                    loss = tf.losses.mean_squared_error(target, self.exp_train_model(state))
                grads = tape.gradient(loss, self.exp_train_model.variables)
                optimizer.apply_gradients(zip(grads, self.exp_train_model.variables), global_step=tf.train.get_or_create_global_step())

            #create new training set, and send old one to garbage collection
            trajectory.gaes = self._normalize_gaes(trajectory.gaes)
            dataset = tf.data.Dataset.from_tensor_slices((trajectory.states, trajectory.rews, trajectory.old_act_probs, trajectory.actions, trajectory.gaes))
            dataset = dataset.shuffle(100).batch(self.ppo_batch_size)

            #update policy and value nets
            p_optimizer = tf.train.AdamOptimizer(learning_rate=self.ppo_lr)
            v_optimizer = tf.train.AdamOptimizer(learning_rate=self.ppo_lr)
            for (batch, (state, rew, old_act_prob, action, gae)) in enumerate(dataset.take(self.ppo_net_opt_steps)):
                with tf.GradientTape(persistent=True) as tape:
                    row_idxs = tf.range(action.shape[0], dtype=tf.int64)
                    action = tf.stack([row_idxs, tf.squeeze(action)], axis=1)
                    features = self.vis_model(state)
                    new_act_probs = self.policy_model(features)
                    new_act_prob = tf.gather_nd(new_act_probs, action)
                    val_e = self.val_model_e(features)
                    val_i = self.val_model_i(features)
                    val = val_e + val_i
                    ratio = tf.exp(tf.log(new_act_prob) - tf.log(old_act_prob))
                    min_gae = tf.where(gae>0, (1+self.clip_value)*gae, (1-self.clip_value)*gae)
                    p_loss = -tf.reduce_mean(tf.minimum(ratio * gae, min_gae))
                    v_loss = tf.reduce_mean(tf.square(rew - val))

                #update vision model and policy head based on policy loss function
                variables = self.vis_model.variables + self.policy_model.variables
                grads = tape.gradient(p_loss, variables)
                p_optimizer.apply_gradients(zip(grads, variables))

                #update vision model and value heads based on value loss function
                variables = self.vis_model.variables + self.val_model_e.variables + self.val_model_i.variables
                grads = tape.gradient(v_loss, variables)
                v_optimizer.apply_gradients(zip(grads, variables))
                del tape

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
        return [self.vis_model.weights,
                self.policy_model.weights,
                self.val_model_e.weights,
                self.val_model_i.weights,
                self.exp_target_model.weights,
                self.exp_train_model.weights]

    @weights.setter
    def weights(self, new_weights):
        self.vis_model.weights = new_weights[0]
        self.policy_model.weights = new_weights[1]
        self.val_model_e.weights = new_weights[2]
        self.val_model_i.weights = new_weights[3]
        self.exp_target_model.weights = new_weights[4]
        self.exp_target_model.weights = new_weights[5]

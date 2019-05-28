from copy import deepcopy
import os
from operator import add

import numpy as np
import tensorflow as tf
from mpi4py import MPI

from supersonic import environment, utils, models, logger


class PPOAgent:
    """
    Basic version of Proximal Policy Optimization (Clip) with exploration by Random Network Distillation.
    """
    def __init__(
        self,
        env_id,
        exp_lr=0.001,
        ppo_lr=0.0001,
        vis_model="NatureVision",
        policy_model="VanillaPolicy",
        val_model="VanillaValue",
        exp_target_model="ExplorationTarget",
        exp_train_model="ExplorationTrain",
        exp_epochs=4,
        gamma_i=0.99,
        gamma_e=0.999,
        log_dir=None,
        rollout_length=128,
        ppo_epochs=4,
        e_rew_coeff=2.0,
        i_rew_coeff=1.0,
        vf_coeff=0.5,
        exp_train_prop=0.25,
        lam=0.95,
        exp_batch_size=32,
        ppo_batch_size=32,
        ppo_clip_value=0.1,
        checkpoint_interval=500,
        minkl=None,
        entropy_coeff=.0003,
        random_rollouts=25,
        max_grad_norm=.25,
    ):

        tf.enable_eager_execution()
        tf.logging.set_verbosity(tf.logging.FATAL)
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.env = environment.auto_env(env_id)
        self.random_rollouts = random_rollouts

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
        self.global_step = tf.train.create_global_step()
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

        self.i_rew_running_stats = utils.RunningStats()

        if self.rank == 0:
            self.log_dir = os.path.join("logs", log_dir) if log_dir else "logs/"
            self.logger = logger.Logger(self.log_dir)
            self._log_episode_num = 0
            self._reset_stats()
    
    def _update_exp_model(self, grads):
        variables = self.exp_train_model.variables
        self.exp_optimizer.apply_gradients(
            zip(grads, variables), global_step=self.global_step,
        )
    
    def _update_ppo_models(self, grads):
        variables = (
                self.vis_model.variables
                + self.policy_model.variables
                + self.val_model_e.variables
                + self.val_model_i.variables
        )
        self.ppo_optimizer.apply_gradients(zip(grads, variables), global_step=self.global_step)

    def train(self, rollouts, render=0):
        # calibrate i_rew normalizaiton stats using random rollouts
        for random_rollout in range(self.random_rollouts):
            trajectory = self._rollout(self.rollout_length, render=False)
            if self.rank == 0:
                # update exploration nets to bring the loss to normal levels once ppo begins to train
                exp_grads, _ = self._get_grads(trajectory, exp_only=True)
                self._update_exp_model(exp_grads)
        # share exploration net with rest of workers
        self.weights = self.comm.bcast(self.weights, root=0)
        self.comm.barrier()
        if self.rank == 0:
            progbar = tf.keras.utils.Progbar(rollouts)
        for rollout in range(rollouts):
            #collect a rollout
            trajectory = self._rollout(
                self.rollout_length,
                render=True if self.rank <= render else False,
            )
            self.comm.barrier()
            # collect grads from workers
            grads_exp, grads_ppo = self._get_grads(trajectory)
            gather_grads_exp = self.comm.gather(grads_exp, 0)
            gather_grads_ppo = self.comm.gather(grads_ppo, 0)
            if self.rank == 0:
                #average grads and apply update
                sum_grads_exp = [sum(grad) for grad in zip(*gather_grads_exp)]
                avg_grads_exp = [sum_grad / self.comm.size for sum_grad in sum_grads_exp]
                self._update_exp_model(avg_grads_exp)
                sum_grads_ppo = [sum(grad) for grad in zip(*gather_grads_ppo)]
                avg_grads_ppo = [sum_grad / self.comm.size for sum_grad in sum_grads_ppo]
                self._update_ppo_models(avg_grads_ppo)
            self.comm.barrier()
            # broadcast new weights to workers
            self.weights = self.comm.bcast(self.weights, root=0)
            if self.stop:
                break # if early stopping activated
            if self.rank == 0:
                progbar.update(rollout + 1)
                if rollout % self.checkpoint_interval == 0:
                    self._checkpoint(rollout)
            self.comm.barrier()
        self.save_weights("final")

    def test(self, episodes, max_ep_steps=4500, render=False, stochastic=True):
        try:
            self.env.current_max_steps = max_ep_steps
        except:
            pass
        cum_rew = 0
        for ep in range(episodes):
            step = 0
            obs, rew, done, info = self.env.reset(), 0, False, {}
            while step < max_ep_steps and not done:
                action = self._choose_action(obs, stochastic=stochastic)
                obs, rew, done, info = self.env.step(action)
                if render:
                    self.env.render()
                cum_rew += rew
                step += 1
        return cum_rew

    def _rollout(self, steps, render=False):
        """
        Step through the environment using current policy. Calculate all the metrics needed by 
        update_models() and save it to a util.Trajectory object
        """
        if not self.most_recent_step[2]:  # if not done
            obs, e_rew, done, info = self.most_recent_step
        else:
            obs, e_rew, done, info = self.env.reset(), 0, False, {}
        step = 0
        trajectory = utils.Trajectory(
            self.rollout_length, self.i_rew_running_stats
        )
        while step < steps:
            if done:
                obs, e_rew, done, info = self.env.reset(), 0, False, {}
            action_prob, action, val_e, val_i = self._choose_action_get_value(obs)
            val_e, val_i = (val_e, val_i) if not done else (0, val_i)
            obs2, e_rew, done, info = self.env.step(action)
            if render:
                self.env.render()
            i_rew, exp_target = self._calc_intrinsic_reward(obs)
            trajectory.add(
                obs, e_rew, i_rew, exp_target, (action_prob, action), val_e, val_i, done
            )
            if self.env_is_sonic and self.rank == 0:
                self._update_ep_stats(action, e_rew, i_rew, done, info)
            obs = obs2
            step += 1
        *_, last_val_e, last_val_i = (
            self._choose_action_get_value(obs) if not done else (0, 0, 0, 0)
        )
        self.most_recent_step = (obs, e_rew, done, info)
        trajectory.end_trajectory(
            self.e_rew_coeff,
            self.i_rew_coeff,
            self.gamma_i,
            self.gamma_e,
            self.lam,
            last_val_i,
            last_val_e,
        )
        return trajectory

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
        if info["lives"] < self._log_current_lives:
            self._log_current_lives -= 1
            current_pos = (info["screen_x"], info["screen_y"])
            self._log_death_coords.append(current_pos)
            self._log_furthest_point = max(self._log_furthest_point, current_pos)
        if done:
            if "FORCED EXIT" in info:
                self._log_death_coords = (info["screen_x"], info["screen_y"])
                self._log_furthest_point = self._log_death_coords
            episode_dict = {
                "episode_num": self._log_episode_num,
                "death_coords": self._log_death_coords,
                "training_steps": self._log_training_steps,
                "max_x": self._log_furthest_point[0],
                "score": info["score"],
                "external_reward": self._log_cum_rew_e,
                "internal_reward": self._log_cum_rew_i,
                "action_count": self._log_action_count,
            }
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
        action_probs = np.squeeze(self.policy_model(features)) + 1e-8
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
        return rew, target

    def _checkpoint(self, rollout_num):
        save_path = f"weights/{os.path.basename(self.log_dir)}/checkpoint_{rollout_num}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_weights(save_path)

    def _get_grads(self, trajectory, exp_only=False):
        # get exploration network grads
        exp_train_samples = int(self.rollout_length * self.exp_train_prop)
        idxs = np.random.choice(trajectory.states.shape[0], exp_train_samples, replace=False)
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                np.take(trajectory.states, idxs, axis=0),
                np.take(trajectory.exp_targets, idxs, axis=0),
            )
        )
        dataset = (
            dataset.shuffle(exp_train_samples + 1)
            .repeat(self.exp_epochs)
            .batch(self.exp_batch_size)
        )

        for (batch, (state, target)) in enumerate(dataset):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.square(target - self.exp_train_model(state)))
            exp_grads = tape.gradient(loss, self.exp_train_model.variables)
            if self.max_grad_norm:
                exp_grads, _ = tf.clip_by_global_norm(exp_grads, self.max_grad_norm)

        if exp_only:
            return exp_grads, None

        # create new training set
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                trajectory.states,
                trajectory.rets_e,
                trajectory.rets_i,
                trajectory.old_act_probs,
                trajectory.actions,
                trajectory.gaes,
            )
        )
        dataset = (
            dataset.shuffle(trajectory.states.shape[0] + 1)
            .repeat(self.ppo_epochs)
            .batch(self.ppo_batch_size)
        )
        # get ppo network grads
        for (batch, (state, e_ret, i_ret, old_act_prob, action, gae)) in enumerate(dataset):
            with tf.GradientTape() as tape:
                row_idxs = tf.range(action.shape[0], dtype=tf.int64)
                action = tf.stack([row_idxs, tf.squeeze(action)], axis=1)
                features = self.vis_model(state)
                new_act_probs = self.policy_model(features) + 1e-8
                new_act_prob = tf.log(tf.gather_nd(new_act_probs, action))
                old_act_prob = tf.squeeze(old_act_prob)

                val_e = self.val_model_e(features)
                val_e_loss = tf.reduce_mean(tf.square(e_ret - val_e))
                val_i = self.val_model_i(features)
                val_i_loss = tf.reduce_mean(tf.square(i_ret - val_i))
                v_loss = val_e_loss + val_i_loss

                ratio = tf.exp(new_act_prob - old_act_prob)
                min_gae = tf.where(
                    gae >= 0, (1 + self.clip_value) * gae, (1 - self.clip_value) * gae
                )
                p_loss = -tf.reduce_mean(tf.minimum(ratio * gae, min_gae))
                entropy = tf.reduce_mean(
                    -tf.reduce_sum(new_act_probs * tf.log(new_act_probs), axis=1)
                )
                loss = p_loss + self.vf_coeff * v_loss - self.entropy_coeff * entropy

            variables = (
                self.vis_model.variables
                + self.policy_model.variables
                + self.val_model_e.variables
                + self.val_model_i.variables
            )
            ppo_grads = tape.gradient(loss, variables)
            if self.max_grad_norm:
                ppo_grads, _ = tf.clip_by_global_norm(ppo_grads, self.max_grad_norm)

            approxkl = tf.reduce_mean(old_act_prob - new_act_prob)
            if self.minkl and approxkl < self.minkl:
                self._checkpoint("earlyStopped")
                self.stop = True
            
            return exp_grads, ppo_grads

    def save_weights(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.vis_model.save_weights(os.path.join(path, "vis_model"))
        self.policy_model.save_weights(os.path.join(path, "pol_model"))
        self.val_model_e.save_weights(os.path.join(path, "val_model_e"))
        self.val_model_i.save_weights(os.path.join(path, "val_model_i"))
        self.exp_train_model.save_weights(os.path.join(path, "exp_train_model"))
        self.exp_target_model.save_weights(os.path.join(path, "exp_target_model"))

    def load_weights(self, path):
        self.vis_model.load_weights(os.path.join(path, "vis_model"))
        self.policy_model.load_weights(os.path.join(path, "pol_model"))
        self.val_model_e.load_weights(os.path.join(path, "val_model_e"))
        self.val_model_i.load_weights(os.path.join(path, "val_model_i"))
        self.exp_train_model.load_weights(os.path.join(path, "exp_train_model"))
        self.exp_target_model.load_weights(os.path.join(path, "exp_target_model"))

    @property
    def weights(self):
        return [
            self.vis_model.get_weights(),
            self.policy_model.get_weights(),
            self.val_model_e.get_weights(),
            self.val_model_i.get_weights(),
            self.exp_target_model.get_weights(),
            self.exp_train_model.get_weights(),
        ]

    @weights.setter
    def weights(self, new_weights):
        self.vis_model.set_weights(new_weights[0])
        self.policy_model.set_weights(new_weights[1])
        self.val_model_e.set_weights(new_weights[2])
        self.val_model_i.set_weights(new_weights[3])
        self.exp_target_model.set_weights(new_weights[4])
        self.exp_train_model.set_weights(new_weights[5])

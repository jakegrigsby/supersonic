import numpy as np
import tensorflow as tf

import environment
import utils

class BaseAgent:
    """
    The version of PPO used for meta learning could be different than sprinting.
    """
    def __init__(self, env_id, **kwargs):
        """lot more params to come so not dealing with that for now"""
        self.env = environemnt.auto_env(env_id)

        self.exp_lr = exp_lr
        self.policy_lr = policy_lr
        self.val_lr = val_lr

        self.vis_model = vis_model()
        self.policy_model = policy_model(env.action_space.shape)
        self.val_model = val_model()
        self.exp_target_model = exp_target_model()
        self.exp_train_model = exp_train_model()

        self.exp_net_opt_steps = exp_net_opt_steps

    def train(self, rollouts, device):
        for _ in range(rollouts):
            trajectory = self.rollout(steps)
            self.update_models(trajectory, device)

    def test(self, episodes, max_ep_steps=4500):
        for ep in range(episodes):
            step = 0
            obs, rew, done, info = self.env.reset()
            while step < max_ep_steps:
                action = self.choose_action(obs)
                obs, rew, done, info = self.env.step(action)

    def rollout(steps):
        """
        Step through the environment using current policy. Calculate all the metrics needed by update_models() and save it to a util.Trajectory object
        """
        raise NotImplementedError()

    def choose_action(self, obs):
        features = self.vis_model(obs)
        actions = self.policy_model(features)
        return np.argmax(actions)

    def calc_intrinsic_reward(self, state):
        target = self.exp_target_model(state)
        pred = self.exp_train_model(state)
        rew = np.abs(target - pred)
        return rew, target #save targets to trajectory to avoid another call to the exp_target_model network during update_models()


    def update_models(self, trajectory, device):
        with tf.device(device):
            #create new training set
            dataset = tf.data.Dataset.from_tensor_slices((trajectory.states, trajectory.exp_targets))
            dataset.shuffle(100)

            #update exploration net
            optimizer = tf.train.AdamOptimizer(learning_rate=self.exp_lr)
            step = 0
            for (batch, (state, target)) in enumerate(exp_dataset.take(64)):
                if step > self.exp_net_opt_steps: break
                with tf.GradientTape() as tape:
                    loss = tf.math.abs(target - self.exp_train_model(state))
                grads = tape.gradient(loss, self.exp_train_model.variables)
                optimizer.apply_gradients(zip(grads, self.exp_train_model.variables), global_step=tf.train.get_or_create_global_step())
                step += 1

            #create new training set, and send old one to garbage collection
            dataset = tf.data.Dataset.from_tensor_slices((trajectory.states, trajectory.rews, trajectory.old_act_probs, trajectory.gaes))
            dataset.shuffle(100)
            
            #update policy and value nets
            p_optimizer = tf.train.AdamOptimizer(learning_rate=self.policy_lr)
            v_optimizer = tf.train.AdamOptimizer(learning_rate=self.val_lr)
            step = 0
            for (batch, (state, rew, old_act_prob, gae)) in enumerate(dataset.take(64)):
                if step > self.policy_net_opt_steps and step > self.val_net_opt_steps: break
                with tf.GradientTape(persistent=True) as tape:
                    features = self.viz_model(state)
                    new_act_probs = self.policy_model(features)
                    val = self.value_model(features)
                    ratios = tf.exp(tf.log(new_act_probs) - tf.log(old_act_probs))
                    min_gae = tf.where(gae>0, (1+self.clip_value)*gae, (1-self.clip_value)*gae)
                    p_loss = -tf.reduce_mean(tf.minimum(ratio * gae, min_gae))
                    v_loss = tf.reduce_mean(tf.square(rew - val))

                grads = tape.gradient(p_loss, [self.vis_model.variables, self.policy_model.variables])
                p_optimizer.apply_gradients(zip(grads, [self.vis_model.variables, self.policy_model.variables]), global_step=tf.train.get_or_create_global_step())
                grads = tape.gradient(v_loss, [self.viz_model.variables, self.val_model.variables])
                v_optimizer.apply_gradients(zip(grads, [self.vis_model.variables, self.val_model.variables]))
                del tape
                step += 1 









import numpy as np
import tensorflow as tf

import environment
import utils
import models

def ppo_agent(env_id, hyp_dict, log_dir, name):
    """
    TODO: This needs updating

    When tuning hyperparameters, it'll be easier to create and modify dictionaries, which can be passed between
    MPI processes more efficiently. This function takes a dictionary describing the agent's params, as well as a
    string representing the environment and a log dir, and creates a BaseAgent.
    """
    x = hyp_dict
    return BaseAgent(env_id,
                    exp_lr = x['exp_lr']
                    policy_lr = x['policy_lr']
                    val_lr = x['val_lr']
                    vis_model = models.get_model(x['vis_model']),
                    policy_model = models.get_model(x['policy_model']),
                    val_model = models.get_model(x['val_model']),
                    exp_target_model = models.get_model(x['exp_target_model']),
                    exp_train_model = models.get_model(x['exp_train_model']),
                    exp_net_opt_steps = x['exp_net_opt_steps'],
                    gamma = x['gamma'],
                    lam = x['lam'],
                    log_dir = log_dir
                    name = name
                    )

class BaseAgent:
    """
    Basic version of Proximal Policy Optimization (Clip) with exploration by Random Network Distillation.

    Needs a lot of testing and probably debugging. I wrote up a quick outline of the training loop.
    
    Also TODO is all the calculations of metric we want to log, and the actual logging at the end of each episode.
    """
    def __init__(self, env_id, exp_lr=.001, ppo_lr=.001, vis_model='NatureVision', policy_model='NaturePolicy', val_model='VanillaValue',
                    exp_target_model='NatureVision', exp_train_model='NatureVision', exp_net_opt_steps=None, gamma_i=.99, gamma_e=.999, log_dir=None,
                    rollout_length=128, ppo_net_opt_steps=None, e_rew_coeff=2., i_rew_coeff=1., exp_train_prop=.25):

        self.env = environemnt.auto_env(env_id)

        self.exp_lr = exp_lr
        self.ppo_lr = ppo_lr

        self.vis_model = models.get_model(vis_model)
        self.policy_model = models.get_model(policy_model)
        self.val_model_e = models.get_model(val_model)
        self.val_model_i = models.get_model(val_model)
        self.exp_target_model = models.get_model(exp_target_model)
        self.exp_train_model = models.get_model(exp_train_model)

        self.exp_net_opt_steps = exp_net_opt_steps if exp_net_opt_steps else int(4 * (rollout_length * exp_train_prop))
        self.exp_train_prop = exp_train_prop
        self.ppo_net_opt_steps = ppo_net_opt_steps if ppo_net_opt_steps else 4 * rollout_length

        self.gamma_i = gamma_i
        self.gamma_e = gamma_e
        self.lam = lam
        self.e_rew_coeff = e_rew_coeff
        self.i_rew_coeff = i_rew_coeff

        self.rollout_length = rollout_length

        if not log_dir:
            # there is probably a better way to name these
            id = uuid.uuid4()
            log_dir = os.path.join('logs', id)
            os.mkdir(log_dir)
        self.log_dir = log_dir

    def train(self, rollouts, device):
        for _ in range(rollouts):
            trajectory = self.rollout(self.rollout_length)
            self.update_models(trajectory, device)

    def test(self, episodes, max_ep_steps=4500):
    """
    Test the agent's performance.
    """
        for ep in range(episodes):
            step = 0
            obs, rew, done, info = self.env.reset()
            while step < max_ep_steps and not done:
                action = self.choose_action(obs)
                obs, rew, done, info = self.env.step(action)

    def rollout(steps):
        """
        Step through the environment using current policy. Calculate all the metrics needed by update_models() and save it to a util.Trajectory object
        """
        obs, e_rew, done, info = self.env.reset()
        step = 0
        trajectory = utils.Trajectory()
        while step < steps and not done:
            action_probs, val_e, val_i = self.choose_action_get_value(obs)
            action = np.argmax(action_probs)
            obs, e_rew, done, info = self.env.step(action)
            i_rew, exp_target = self.calc_intrinsic_reward(obs)        
            trajectory.add(obs, e_rew, i_rew, exp_target, action_probs, val_e, val_i) 
            step += 1
        trajectory.end_trajectory(self.gamma, self.lam, self.i_rew_coeff, self.e_rew_coeff)
        return trajectory

    def choose_action(self, obs):
        """
        Choose an action based on the current observation. Saves computation by not running the value net,
        which makes it a good choice for testing the agent.
        """
        features = self.vis_model(obs)
        actions = self.policy_model(features)
        return np.argmax(actions)

    def choose_action_get_value(self, obs):
        """
        Run the entire network and return the action probability distribution (not just the chosen action) as well as the values
        from the value net. Used during training -  when more information is needed.
        """
        features = self.vis_model(obs)
        action_probs = self.policy_model(features)
        val_e = self.val_model_e(features)
        val_i = self.val_model_i(features)
        return action_probs, val_e, val_i

    def calc_intrinsic_reward(self, state):
        """
        reward as described in Random Network Distillation
        """
        target = self.exp_target_model(state)
        pred = self.exp_train_model(state)
        rew = np.square(np.subtract(target, pred)).mean()
        return rew, target #save targets to trajectory to avoid another call to the exp_target_model network during update_models()


    def update_models(self, trajectory, device):
    """
    Update the vision, policy, value, and exploration (RND) networks based on a utils.Trajectory object generated by a rollout.
    Should be able specify device so that multiple agents can be efficiently trained on the same (multi-gpu) machine.
    """
        with tf.device(device):
            #create new training set
            exp_train_samples = int(self.rollout_length * self.exp_train_prop)
            idxs = np.random.choice(trajectory.states.shape[0], exp_train_samples, replace=False)
            #another option: idxs = np.random.choice([False, True], trajectory.states.shape[0], p=[.75, .25])
            dataset = tf.data.Dataset.from_tensor_slices((trajectory.states[idxs], trajectory.exp_targets[idxs]))
            dataset.shuffle(100)

            #update exploration net
            optimizer = tf.train.AdamOptimizer(learning_rate=self.exp_lr)
            step = 0
            for (batch, (state, target)) in enumerate(exp_dataset.take(64)):
                if step > self.exp_net_opt_steps: break
                with tf.GradientTape() as tape:
                    loss = tf.losses.mean_squared_error(target, self.exp_train_model(state))
                grads = tape.gradient(loss, self.exp_train_model.variables)
                optimizer.apply_gradients(zip(grads, self.exp_train_model.variables), global_step=tf.train.get_or_create_global_step())
                step += 1

            #create new training set, and send old one to garbage collection
            dataset = tf.data.Dataset.from_tensor_slices((trajectory.states, trajectory.rews, trajectory.old_act_probs, trajectory.gaes))
            dataset.shuffle(100)
            
            #update policy and value nets
            p_optimizer = tf.train.AdamOptimizer(learning_rate=self.ppo_lr)
            v_optimizer = tf.train.AdamOptimizer(learning_rate=self.ppo_lr)
            step = 0
            for (batch, (state, rew, old_act_prob, gae)) in enumerate(dataset.take(64)):
                if step > self.ppo_net_opt_steps: break
                with tf.GradientTape(persistent=True) as tape:
                    features = self.vis_model(state)
                    new_act_probs = self.policy_model(features)
                    val_e = self.val_model_e(features)
                    val_i = self.val_model_i(features)
                    val = val_e + val_i
                    ratios = tf.exp(tf.log(new_act_probs) - tf.log(old_act_probs))
                    min_gae = tf.where(gae>0, (1+self.clip_value)*gae, (1-self.clip_value)*gae)
                    p_loss = -tf.reduce_mean(tf.minimum(ratio * gae, min_gae))
                    v_loss = tf.reduce_mean(tf.square(rew + self.gamma *  - val))

                grads = tape.gradient(p_loss, [self.vis_model.variables, self.policy_model.variables])
                p_optimizer.apply_gradients(zip(grads, [self.vis_model.variables, self.policy_model.variables]), global_step=tf.train.get_or_create_global_step())
                grads = tape.gradient(v_loss, [self.viz_model.variables, self.val_model.variables])
                v_optimizer.apply_gradients(zip(grads, [self.vis_model.variables, self.val_model_e.variables, self.val_model_i.variables]))
                del tape
                step += 1 
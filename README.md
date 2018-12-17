# Sonic Transfer Learning Project Roadmap/Planning

## Long term roadmap:
1. Get a single agent training on Rivanna.
2. Get as many agents training in parallel on one node as possible.
3. Figuring out networking between those crammed nodes.
4. Meta learning
5. PPO alg desing (and CNN)
6. Hyperparameter optimization



I think we can get up to step 3 over break.

### 1) Environments
infrastructure for assembling environment from level id string. Would be nice if it could switch between regular gym Atari and gym retro (Sonic).

goal:
```python
		env = build_env(env_id)
		
		or

		env = Environment(env_id, sticky_frames=True, ram_acc=True, other flags...)
```
Where `Environment` can determine if the env you're looking for is in the ALE or gym retro, then return an instance of the right env type. `build_env()` could be a shortcut
that returns either the ALE env with default settings (`gym.make(env_id)`) or returns the exact setting we're using for Sonic.

##### Sonic Environment Wrappers:
wrappers are used in gym to modify environments. They all take an env as a param. Then each method adds the new stuff you need before returning the same function on self.env.
See the `gym.wrappers` or examples in random network distillation `atari_wrappers`. That way each wrapper pushes the step() command down the env stack before the result bubbles
back to the top.

```python
env = retro.make(lvl_id)
env = StickyActionEnv(env)	#sticky frames from paper and usual ALE stuff. 25% of the frames accept no new action and repeat the old one (not deterministic)
env = MaxAndSkipEnv(env)	
env = SonicInfoWrapper(env) 	#access ram for analysis purposes. How far into the level are we, what's the score, lives time... stuff like that.
#maybe more idk
```

##### other notes:
- accessing RAM will need to be done differently for ALE vs retro. reg gym envs have `ale.getRAM()`. retro envs have `data` that return a dict of RAM values laid out in `data.json`
- in most retro envs you need to define `scenario.json`, `data.json` which points out values in RAM and how to calc reward func, when game is over and some other things. See retro docs.
But I think the retro contest already did this stuff for us for Sonic. there's a `contest.json` in each Sonic game data foler. Just need to make sure that's the one being used.


### 2) Prototype PPO
We need to throw together a version of PPO that works. It can just be an open source, vanilla version with frozen hyperparameters wrapped in an API we come up with (for now). But
we need to come up w that api so we can build everything else. 
	

thoughts...
```python
agent = Agent(hyperparams) #creates ppo object
agent.model = 'pretrained_weights.hd5f'
agent.learn(env, steps)
updated_weights = agent.model
```
that's going to take some more planning


### 3) Logging/Recording
- Logging all the useful data (reward, step, stuff from ram like x pos, action, anything else we can think of)
- When to log, how best to store it
- recording clips of gameplay automatically
	- how do you determine when to start and stop recording?
		- maybe when it's been stuck for a while but gets over an obstacle
			- buffer so we can decide to save footage after it's already happened (`xbox record that`)
		- when it's reaching an area it never has before
	- retro envs have `record_movie()` and `stop_record()` already. the `Environment` class can copy those to use for ALE stuff too. (reg gym has its own record system but this one
	seems simpler).


### 4) Visualizing Data
- reading in the log files for every training run... or every training run on a certain level... or a whole bunch of other stuff... and making a database.
- then visualizing the data we're looking for (GUI yikes).
	-matplotlib
- we're going to need to be able to do live monitoring at some point, because the meta learning and hyperparam opt training runs we'll be doing eventually are going to be so long.
we should probably plan for that the first time so we don't need to redo it.
- our laptops grab the log files, build the database, display live graphs/recorded video clips, update every minute or so. 

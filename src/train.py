from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
import torch
import torch.nn as nn
import pickle
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from torch.distributions.categorical import Categorical
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ProjectAgent:
    def __init__(self, config):
        env = gym.make(config.env_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.config.batch_size = int(config.num_steps)
        self.config.minibatch_size = int(config.batch_size // config.num_minibatches)
        self.config.num_iterations = config.total_timesteps // config.batch_size
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(env.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        ).to(self.device)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(env.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, env.single_action_space.n), std=0.01),
        ).to(self.device)
        self.env = env
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=config.learning_rate, eps=1e-5)
    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    
    def act(self, observation, use_random=False):
        logits = self.actor(observation)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action

    def save(self, path):
        serialized= {"actor": self.actor, "critic":self.critic, "config":self.config}
        with open(path+'saved.pkl', 'wb') as f:  # open a text file
            pickle.dump(serialized, f) # serialize the list
    def load(self):
        with open('saved.pkl', 'wb') as f:  # open a text file
            saved = pickle.load( f) # serialize the list
        self.__init__(saved.config)
        self.actor = saved.actor
        self.critic = saved.critic
        try : 
            x,_ = self.env.reset()
            self.actor(x)
        except : 
            raise "Actor incompatible with environnement"
    def train(self): 
        obs = torch.zeros(self.config.num_steps ,env.single_observation_space.shape).to(self.device)
        actions = torch.zeros(self.config.num_steps , env.single_action_space.shape).to(self.device)
        logprobs = torch.zeros(self.config.num_steps).to(self.device)
        rewards = torch.zeros(self.config.num_steps).to(self.device)
        dones = torch.zeros(self.config.num_steps).to(self.device)
        values = torch.zeros(self.config.num_steps).to(self.device)
        global_step = 0
        next_obs, _ = env.reset(seed=self.config.seed)
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(1).to(self.device)
        with tqdm(range(1, self.config.num_iterations + 1)) as pbar : 
            for iteration in pbar:
                for step in range(0, self.config.num_steps):
                    global_step += 1
                    obs[step] = next_obs
                    dones[step] = next_done
                    # ALGO LOGIC: action logic
                    with torch.no_grad():
                        action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                        values[step] = value.flatten()
                    actions[step] = action
                    logprobs[step] = logprob
                    # TRY NOT TO MODIFY: execute the game and log data.
                    next_obs, reward, terminations, truncations, infos = self.env.step(action.cpu().numpy())
                    next_done = np.logical_or(terminations, truncations)
                    rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                    next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(next_done).to(self.device)
                    # bootstrap value if not done
                with torch.no_grad():
                    next_value = self.agent.get_value(next_obs).reshape(1, -1)
                    advantages = torch.zeros_like(rewards).to(self.device)
                    lastgaelam = 0
                    for t in reversed(range(self.config.num_steps)):
                        if t == self.config.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        delta = rewards[t] + self.config.gamma * nextvalues * nextnonterminal - values[t]
                        advantages[t] = lastgaelam = delta + self.config.gamma * self.config.gae_lambda * nextnonterminal * lastgaelam
                    returns = advantages + values
                # flatten the batch, normalement ici pas besoin
                b_obs = obs.reshape((-1,) + self.env.single_observation_space.shape)
                b_logprobs = logprobs.reshape(-1)
                b_actions = actions.reshape((-1,) + self.env.single_action_space.shape)
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1)
                pbar.set_postfix(avg_returns = np.mean(returns))
                b_values = values.reshape(-1)
                # Optimizing the policy and value network
                b_inds = np.arange(self.config.batch_size)
                clipfracs = []
                for epoch in range(self.config.update_epochs):
                    np.random.shuffle(b_inds)
                    for start in range(0, self.config.batch_size, self.config.minibatch_size):
                        end = start + self.config.minibatch_size
                        mb_inds = b_inds[start:end]

                        _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()

                        with torch.no_grad():
                            # calculate approx_kl http://joschu.net/blog/kl-approx.html
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clipfracs += [((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item()]

                        mb_advantages = b_advantages[mb_inds]
                        if self.config.norm_adv:
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                        # Policy loss
                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef)
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                        # Value loss
                        newvalue = newvalue.view(-1)
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.config.clip_coef,
                            self.config.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()

                        entropy_loss = entropy.mean()
                        loss = pg_loss - self.config.ent_coef * entropy_loss + v_loss * self.config.vf_coef

                        self.optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
                        self.optimizer.step()

                    if self.config.target_kl is not None and approx_kl > self.config.target_kl:
                        break
 

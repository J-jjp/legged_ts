

import torch
import numpy as np

from rsl_rl.utils import split_and_pad_trajectories

class RolloutStorage_ts:
    class Transition_ts:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.student_action = None

            self.dones = None
            self.teacher_action = None

            self.hidden_states = None
        
        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, critic_observations, actions_shape, device='cpu'):

        self.device = device

        self.obs_shape = obs_shape
        self.critic_observations = critic_observations
        self.actions_shape = actions_shape

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)

        self.critic_observations = torch.zeros(num_transitions_per_env, num_envs, *critic_observations, device=self.device)


        self.student_action = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
     
        self.teacher_action = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

   

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # rnn
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.step = 0


    def add_transitions_ts(self, transition_ts: Transition_ts):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition_ts.observations)
        self.critic_observations[self.step].copy_(transition_ts.critic_observations)
        self.student_action[self.step].copy_(transition_ts.student_action)

        self.dones[self.step].copy_(transition_ts.dones.view(-1, 1))
        self.teacher_action[self.step].copy_(transition_ts.teacher_action)


        self._save_hidden_states(transition_ts.hidden_states)
        self.step += 1
    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states==(None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)

        # initialize if needed 
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))]
            self.saved_hidden_states_c = [torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])


    def clear(self):
        self.step = 0

    def mini_batch_generator_ts(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)
        # print(f"batch_size: {batch_size}, mini_batch_size: {mini_batch_size}")
        # print(f"num_mini_batches: {num_mini_batches}, num_epochs: {num_epochs}")
        observations = self.observations.flatten(0, 1)

        critic_observations = self.critic_observations.flatten(0, 1)


        student_action = self.student_action.flatten(0, 1)
        teacher_action = self.teacher_action.flatten(0, 1)
        # print(f"observations shape: {observations.shape}")
        # print(f"critic_observations shape: {critic_observations.shape}")
        # print(f"student_action shape: {student_action.shape}")
        # print(f"teacher_action shape: {teacher_action.shape}")

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):

                start = i*mini_batch_size
                end = (i+1)*mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                student_action_batch = student_action[batch_idx]
                teacher_action_batch = teacher_action[batch_idx]

                yield obs_batch, critic_observations_batch, student_action_batch, teacher_action_batch, \
                       (None, None), None

   
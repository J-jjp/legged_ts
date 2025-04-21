# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic,MLP_Encoder
from rsl_rl.storage import RolloutStorage_ts

class PPO_ts:
    teacher: ActorCritic
    teacher_encoder: MLP_Encoder

    student: ActorCritic
    student_encoder: MLP_Encoder
    def __init__(self,
                 teacher,
                 student,
                 teacher_encoder,
                 student_encoder,
                 num_learning_epochs=5,
                 num_mini_batches=6,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.teacher = teacher
        self.teacher.to(self.device)
        self.student = student
        self.student.to(self.device)
        
        self.teacher_encoder = teacher_encoder
        self.teacher_encoder.to(self.device)
        self.student_encoder = student_encoder
        self.student_encoder.to(self.device)

        self.storage = None # initialized later
        self.optimizer = optim.Adam( list(self.student.parameters()) + list(self.student_encoder.parameters()), lr=learning_rate)

        self.transition = RolloutStorage_ts.Transition_ts()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage_ts(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)


    def act(self, obs, critic_obs):
        
        # Compute the actions and values
        teacher_encoder_out = self.teacher_encoder(critic_obs).detach()
        student_encoder_out =self.student_encoder(obs)

        self.transition.student_action = self.student.act_inference(torch.cat((obs,student_encoder_out),dim=-1)).detach()
        self.transition.teacher_action = self.teacher.act_inference(torch.cat((obs,teacher_encoder_out),dim=-1)).detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.student_action
    
    def process_env_step(self, rewards, dones, infos):
  
        self.transition.dones = dones

        # Record the transition
        self.storage.add_transitions_ts(self.transition)
        self.transition.clear()
        self.student.reset(dones)
    

    def update(self):            
        mean_action_loss = 0
        mean_latent_loss = 0


        generator = self.storage.mini_batch_generator_ts(self.num_mini_batches, self.num_learning_epochs)

        for obs_batch, critic_obs_batch, student_act_batch,teacher_act_batch, hid_states_batch, masks_batch in generator:
            
            
        #     # batch shape: [total traj_length, batch_envs, dim]
            teacher_encoder_out_batch = self.teacher_encoder(critic_obs_batch).detach()
            student_encoder_out_batch =self.student_encoder(obs_batch)
            student_act_batch = self.student.act_inference(torch.cat((obs_batch,student_encoder_out_batch),dim=-1))

        #     # imitate loss
            act_loss = (teacher_act_batch - student_act_batch).pow(2).mean()
            latent_loss = (teacher_encoder_out_batch - student_encoder_out_batch).pow(2).mean()

        #     # Compute total loss.
            loss=latent_loss + act_loss
        #     # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.student.parameters()) + list(self.student_encoder.parameters()), 
                self.max_grad_norm
            )
            self.optimizer.step()
            mean_action_loss += act_loss.item()
            mean_latent_loss += latent_loss.item()
        num_updates = self.num_learning_epochs * self.num_mini_batches

        mean_action_loss /= num_updates
        mean_latent_loss /= num_updates
        self.storage.clear()
        
        return mean_action_loss,mean_latent_loss

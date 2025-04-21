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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import rospy
from sensor_msgs.msg import Joy
joy_cmd = [0.0, 0.0, 0.0]
is_teacher=True
def joy_callback(joy_msg):
    global joy_cmd
    global stop
    global begin
    joy_cmd[0] =  joy_msg.axes[1]
    joy_cmd[1] =  joy_msg.axes[0]
    joy_cmd[2] =  joy_msg.axes[3]  # 横向操作

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.max_init_terrain_level = 0 # 机器人刚开始的难度
    env_cfg.terrain.terrain_length = 8.
    env_cfg.terrain.terrain_width = 8.
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 10
    env_cfg.terrain.terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]

    env_cfg.terrain.curriculum = True
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    priv = env.get_privileged_observations()

    # load policy
    train_cfg.runner.resume = True
    path = "/home/ubuntu/isaac/t_s/legged_gym/logs/rough_a1/Apr21_12-56-07_/model_6500.pt"
    
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env,path=path, name=args.task, args=args, train_cfg=train_cfg,isteacher=is_teacher)
    
    policy = ppo_runner.get_inference_policy(device=env.device)
    encoder = ppo_runner.get_inference_encoder(device=env.device)

    rospy.init_node('play')
    rospy.Subscriber('/joy', Joy, joy_callback, queue_size=10)
    # export policy as a jit module (used to run it from C++)

    if is_teacher is not True:
        if 1:
            export_policy_as_jit(ppo_runner.alg.student,ppo_runner.alg.student_encoder, path)


        print('Exported policy as jit script to: ', path)


    for i in range(10*int(env.max_episode_length)):
        env.commands[:, 0] = joy_cmd[0]
        env.commands[:, 1] = joy_cmd[1]*1
        env.commands[:, 3] = joy_cmd[2]*2
        print("action:",obs.shape,priv.shape)
        if is_teacher is not True:
            encoder_out=encoder(obs).detach()
        else:
            encoder_out=encoder(priv).detach()
        actions = policy(torch.cat((obs.detach(),encoder_out),dim=-1))
        obs, priv, rews, dones, infos = env.step(actions.detach())


if __name__ == '__main__':
    EXPORT_POLICY = True
    args = get_args()
    args.task = "a1"
    play(args)

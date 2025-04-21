
### scripts ###
需要mujoco、ros的包(如麻烦，直接删掉play中的手柄接收)
sim2sim.py --mujoco迁移验证（迁移很奇怪，基座会出现莫名抖动和gym中差异很大）
7000轮次是老师，play中isteacher=true
5000轮次是学生，play中isteacher=false
### rsl_rl ###
1. 在原版基础上teacher加了编码器，输入是obs_critic(obs+privileged_obs),输出32
2. student为后缀加了ts的文件，只做了动作损失和潜在特征损失

(会不会潜在特征编码器需要和动作网络分开更新？推测原因：编码损失一直处于3左右，动作损失为0.17左右)

### leggedgym ###
1. 目前只加了电机偏执、kpkd和strength随机化
2. 修改了常用奖励，经过普通训练验证，各类奖励作用正常可以完成迁移（推测是网络有问题）
3. 将线速度和角速度以及地形观测放在了特权观测中，特权观测加入了历史观测作为obs_critic

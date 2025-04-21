import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
import onnxruntime as ort
from legged_gym.envs import *
import torch
from scipy.spatial.transform import Rotation as R
import rospy
from sensor_msgs.msg import Joy
import os
from legged_gym import LEGGED_MUJOCO_ROOT_DIR

default_dof_pos=[0.1,0.8,-1.5 ,-0.1,0.8,-1.5, 0.1,1,-1.5, -0.1,1,-1.5]#默认角度需要与isacc一致

joy_cmd = [0.0, 0.0, 0.0]
def joy_callback(joy_msg):
    global joy_cmd
    joy_cmd[0] =  joy_msg.axes[1]
    joy_cmd[1] =  joy_msg.axes[0]
    joy_cmd[2] =  joy_msg.axes[3]  # 横向操作

def quat_rotate_inverse(q, v):
    # 确保输入为numpy数组
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c

def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd


    
class Sim2simCfg(A1RoughCfg):

    class sim_config:
        # print("{LEGGED_GYM_ROOT_DIR}",{LEGGED_GYM_ROOT_DIR})

        mujoco_model_path = LEGGED_MUJOCO_ROOT_DIR+'/resources/robots/TOE_dog/xml/scene.xml'
        sim_duration = 60.0
        dt = 0.005 #1Khz底层
        decimation = 4 # 50Hz

    class robot_config:

        kps = np.array(20, dtype=np.double)#PD和isacc内部一致
        kds = np.array(0.5, dtype=np.double)
        tau_limit = 35. * np.ones(12, dtype=np.double)#nm


if __name__ == '__main__':
    rospy.init_node('play')
    rospy.Subscriber('/joy', Joy, joy_callback, queue_size=10)
    print(LEGGED_MUJOCO_ROOT_DIR)

    encoder_model_path =LEGGED_MUJOCO_ROOT_DIR+ "/onnx/encoder_z_input.onnx"
    policy_model_path = LEGGED_MUJOCO_ROOT_DIR+"/onnx/legged.onnx"

    
    encoder = ort.InferenceSession(encoder_model_path, 
                            providers=['CPUExecutionProvider'])
    policy = ort.InferenceSession(policy_model_path, 
                            providers=['CPUExecutionProvider'])
    model = mujoco.MjModel.from_xml_path(Sim2simCfg.sim_config.mujoco_model_path)#载入初始化位置由XML决定
    model.opt.timestep = Sim2simCfg.sim_config.dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    model.opt.gravity = (0, 0, -9.81) 
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((Sim2simCfg.env.num_actions), dtype=np.double)
    action = np.zeros((Sim2simCfg.env.num_actions), dtype=np.double)
    action_flt = np.zeros((Sim2simCfg.env.num_actions), dtype=np.double)
    last_actions = np.zeros((Sim2simCfg.env.num_actions), dtype=np.double)
    lag_buffer = [np.zeros_like(action) for i in range(2+1)]

    hist_obs = deque()
    for _ in range(Sim2simCfg.env.history_len):
        hist_obs.append(np.zeros([1, Sim2simCfg.env.n_proprio], dtype=np.double))
    count_lowlevel = 0

    for _ in tqdm(range(int(Sim2simCfg.sim_config.sim_duration*100/ Sim2simCfg.sim_config.dt)), desc="Simulating..."):

        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data)#从mujoco获取仿真数据
        q = q[-Sim2simCfg.env.num_actions:]
        dq = dq[-Sim2simCfg.env.num_actions:]
        
        # obs_buf =torch.cat((self.base_ang_vel  * self.obs_scales.ang_vel,
        #                     self.base_euler_xyz * self.obs_scales.quat,
        #                     self.commands[:, :3] * self.commands_scale,#xy+航向角速度
        #                     self.reindex((self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos),
        #                     self.reindex(self.dof_vel * self.obs_scales.dof_vel),
        #                     self.action_history_buf[:,-1]),dim=-1)#列表最后一项 [:-1]也就是上一次的

        if 1:
            # 1000hz ->50hz
            foot_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "foot_geom")
            for i in range(data.ncon):
                if data.contact[i].geom1 == foot_geom_id or data.contact[i].geom2 == foot_geom_id:
                    force = data.efc_force[data.contact[i].efc_address]
                    print(f"Foot contact force: {force:.2f} N")
            if count_lowlevel % Sim2simCfg.sim_config.decimation == 0:

                obs = np.zeros([1, Sim2simCfg.env.n_proprio], dtype=np.float32) #1,45
                gravity_vec =  np.array([0., 0., -1.], dtype=np.float32)

                proj_gravity = quat_rotate_inverse(quat,gravity_vec)

                # obs[0, 0] = omega[0] *Sim2simCfg.normalization.obs_scales.ang_vel
                # obs[0, 1] = omega[1] *Sim2simCfg.normalization.obs_scales.ang_vel
                # obs[0, 2] = omega[2] *Sim2simCfg.normalization.obs_scales.ang_vel
                obs[0, 0] = proj_gravity[0] 
                obs[0, 1] = proj_gravity[1] 
                obs[0, 2] = proj_gravity[2] 
                # print("obs:",proj_gravity)
                obs[0, 3] = joy_cmd[0]* Sim2simCfg.normalization.obs_scales.lin_vel*0.8
                obs[0, 4] = joy_cmd[1] * Sim2simCfg.normalization.obs_scales.lin_vel*0.6
                obs[0, 5] = joy_cmd[2] * Sim2simCfg.normalization.obs_scales.ang_vel*0.6
                obs[0, 6:18] = (q-default_dof_pos) * Sim2simCfg.normalization.obs_scales.dof_pos #g关节角度顺序依据修改为样机
                obs[0, 18:30] = dq * Sim2simCfg.normalization.obs_scales.dof_vel
                obs[0, 30:42] = last_actions#上次控制指令
                obs = np.clip(obs, -Sim2simCfg.normalization.clip_observations, Sim2simCfg.normalization.clip_observations)

                # obs_cpu = obs  # 首先将Tensor移动到CPU，然后转换为NumPy数组 
                # for i in range(3):
                #     print("{:.2f}".format(obs_cpu[0][i]))
                # for i in range(3):  
                #     print("{:.2f}".format(obs_cpu[0][i+3]))

                hist_obs.append(obs) #11,1,45
                hist_obs.popleft() #10,1,45

                n_proprio=Sim2simCfg.env.n_proprio
                history_len=Sim2simCfg.env.history_len
                num_z_encoder = Sim2simCfg.env.num_z_encoder
                num_observations= Sim2simCfg.env.num_observations


                encoder_input = np.zeros([1, num_observations], dtype=np.float32)
                encoder_output = np.zeros([1, num_z_encoder], dtype=np.float32) 

                policy_input = np.zeros([1, (num_observations)+num_z_encoder], dtype=np.float32) 
                hist_obs_input = np.zeros([1, history_len*n_proprio], dtype=np.float32)

                encoder_input[0,0:n_proprio]=obs
                for i in range(history_len):#缓存历史观测
                    encoder_input[0, n_proprio +i * n_proprio : n_proprio  +(i + 1) * n_proprio] = hist_obs[i][0, :]
                
                encoder_output_name = encoder.get_outputs()[0].name
                encoder_input_name = encoder.get_inputs()[0].name
                # for i in range(num_observations):
                #     encoder_input[0, i] = 0
                encoder_output = encoder.run([encoder_output_name], {encoder_input_name: encoder_input})[0]
                # print("encoder_output:",encoder_output)
                for i in range(num_observations):
                    policy_input[0, i] = encoder_input[0,i]
                for i in range(num_z_encoder):
                    policy_input[0, i+num_observations] = encoder_output[0,i]
                for i in range(history_len):#缓存历史观测
                    hist_obs_input[0, i * n_proprio : (i + 1) * n_proprio] = hist_obs[i][0, :]
               
                policy_output_name = policy.get_outputs()[0].name
                policy_input_name = policy.get_inputs()[0].name
                
                action[:] = policy.run([policy_output_name], {policy_input_name: policy_input})[0]
                # print("encoder_output:",encoder_output)


                action = np.clip(action, -Sim2simCfg.normalization.clip_actions, Sim2simCfg.normalization.clip_actions)

                # action_flt=_low_pass_action_filter(action,last_actions)
                last_actions=action

                action_flt = action *0.25
                # 直接选择特定索引
                # action_flt[[0, 3, 6, 9]] *= Sim2simCfg.control.hip_scale_reduction

                joint_pos_target = action_flt + default_dof_pos
                target_q=joint_pos_target


            target_dq = np.zeros((Sim2simCfg.env.num_actions), dtype=np.double)
            # Generate PD control
            tau = pd_control(target_q, q, Sim2simCfg.robot_config.kps,
                             target_dq, dq, Sim2simCfg.robot_config.kds)  # Calc torques
            # tau = np.clip(tau, -Sim2simCfg.robot_config.tau_limit, Sim2simCfg.robot_config.tau_limit)  # Clamp torques
            data.ctrl = tau
        else:#air mode test
            obs = np.zeros([1, Sim2simCfg.env.n_proprio], dtype=np.float32) #1,45

            target_q = default_dof_pos
            # target_q[0]=0
            # target_q[1]=3
            # target_q[2]=3
            # target_q[3]=0
            # target_q[4]=3
            # target_q[5]=3     
            #print(eu_ang*57.3)
            target_dq = np.zeros((Sim2simCfg.env.num_actions), dtype=np.double)
            # Generate PD control
            tau = pd_control(target_q, q, Sim2simCfg.robot_config.kps,
                            target_dq, dq, Sim2simCfg.robot_config.kds)  # Calc torques
            tau = np.clip(tau, -Sim2simCfg.robot_config.tau_limit, Sim2simCfg.robot_config.tau_limit)  # Clamp torques
            data.ctrl = tau

        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1

    viewer.close()

    
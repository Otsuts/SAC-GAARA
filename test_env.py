'''
dm_control 环境测试文件
@Author:Otsutsuki_Orance
Date:2022.11.30
usage: ‘python test_env.py’
results:
--------------- TASK1: demonstrating environments ---------------
--------------- TASK1 DONE ---------------
--------------- TASK2: acting in environment and generating gif ---------------
--------------- animation generated ---------------
--------------- animation saved at ./results\test_animation.gif ---------------
--------------- TASK2 DONE ---------------
--------------- TASK3: testing mujoco ---------------
 MuJoCo can convert this quaternion :
[0.5 0.5 0.5 0.5]
To this rotation matrix :
[[0. 0. 1.]
 [1. 0. 0.]
 [0. 1. 0.]]
mjtJoint(mjJNT_FREE=0, mjJNT_BALL=1, mjJNT_SLIDE=2, mjJNT_HINGE=3)
--------------- TASK3 DONE ---------------
'''
from dm_control.mujoco.wrapper.mjbindings import mjlib
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control import suite
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from matplotlib import animation

workspace = './results'


def display_video(frames, framerate=30, name='test_animation.gif'):
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])

    def update(frame):
        im.set_data(frame)
        return [im]

    interval = 1000 / framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    print('-' * 15, 'animation generated', '-' * 15)
    anim.save(os.path.join(workspace, name), writer='pillow')
    print('-' * 15, 'animation saved at', os.path.join(workspace, name), '-' * 15)


# Load one task:
print('-' * 15, 'TASK1: demonstrating environments', '-' * 15)
row, col = 4, len(suite.BENCHMARKING) // 4
# Iterate over a task set:
for index, (domain_name, task_name) in enumerate(suite.BENCHMARKING):
    env = suite.load(domain_name, task_name)
    pixels = env.physics.render()
    plt.subplot(row, col, index + 1)
    plt.imshow(pixels)
plt.show()
print('-' * 15, 'TASK1 DONE', '-' * 15)

# Step through an episode and print out reward, discount and observation.
print('-' * 15, 'TASK2: acting in environment and generating gif', '-' * 15)
env = suite.load(domain_name="cartpole", task_name="swingup")
action_spec = env.action_spec()
frames = []
time_step = env.reset()
frames.append(env.physics.render())
while not time_step.last():
    action = np.random.uniform(action_spec.minimum,
                               action_spec.maximum,
                               size=action_spec.shape)
    time_step = env.step(action)
    frames.append(env.physics.render())
    # print(time_step.reward, time_step.discount, time_step.observation)
all_frames = np.array(frames)

display_video(all_frames, 30)
print('-' * 15, 'TASK2 DONE', '-' * 15)

print('-' * 15, 'TASK3: testing mujoco', '-' * 15)

quat = np.array((.5, .5, .5, .5))
mat = np.zeros((9))
mjlib.mju_quat2Mat(mat, quat)
print(" MuJoCo can convert this quaternion :")
print(quat)
print("To this rotation matrix :")
print(mat.reshape(3, 3))
print(enums.mjtJoint)
print('-' * 15, 'TASK3 DONE', '-' * 15)

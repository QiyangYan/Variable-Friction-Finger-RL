# Learning In-Hand-Manipulation policy for Multi-Shape Objects on Variable Friction Gripper

## Table of Contents
- [Introduction](#introduction)
- [MuJoCo Simulation Env](#mujoco-simulation-env)
- [Only Sliding Video](#only-sliding-video)
- [Only Sliding Training Success Rate](#only-sliding-training-success-rate)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributions and Licensing](#contributions-and-licensing)
- [Reference](#reference)



## Introduction
Welcome to the repository for our project: **Learning In-Hand-Manipulation policy for Multi-Shape Objects on Variable Friction Gripper**. This project addresses the limitations of existing algorithms that control in a model-based manner. Leveraging reinforcement learning, our approach achieves multi-shape control without relying on pre-existing models, enhancing the gripper's versatility and adaptability.

The project utilises the [Gymnasium API](https://gymnasium.farama.org) and employs the [MuJoCo](https://mujoco.readthedocs.io/en/stable/overview.html) physics engine for simulations and being trained purely on CPU. Our work also draws inspiration from the [Gymnasium Robotics](https://robotics.farama.org) framework.

<p align="center">
  <img src="https://github.com/QiyangYan/Variable-Friction-Finger-RL/assets/75078611/77284c8d-a8d7-46dd-aa4c-ed08e54e5a95" alt="Description" width="400">
  <br>
  MuJoCo Simulation Env
</p>

**Published papers on Variable Friction Gripper:**
* [Variable-Friction Finger Surfaces to Enable Within-Hand Manipulation via Gripping and Sliding](https://github.com/QiyangYan/Variable-Friction-Finger-RL/blob/461477be9c9c979466bd3d575dc51b07a4dfb78d/Gripper%20Paper/Variable-Friction%20Finger%20Surfaces%20to%20Enable%20Within-Hand%20Manipulation%20via%20Gripping%20and%20Sliding%20.pdf)
* [Within-Hand Manipulation Planning and Control for Variable Friction Hands](https://github.com/QiyangYan/Variable-Friction-Finger-RL/blob/461477be9c9c979466bd3d575dc51b07a4dfb78d/Gripper%20Paper/Within-Hand%20Manipulation%20Planning%20and%20Control%20for%20Variable%20Friction%20Hands.pdf)

**Previous works on Variable Friction Gripper:**
* [WIHM Variable Friction Finger with Ur5e and Realsense](https://github.com/QiyangYan/WIHM-Variable-Friction-Finger-with-Ur5e-and-Realsense)
* [Variable friction finger](https://github.com/gokul-gokz/Variable_friction_finger)
* [Friction finger gripper RL](https://github.com/gokul-gokz/Friction_finger_gripper_RL)
* [Dextrous In-Hand-Manipulation WPI-MER-LAB](https://github.com/kgnandanwar/Dextrous-In-Hand-Manipulation-WPI-MER-LAB-)
* [WIHM Variable Friction](https://github.com/asahin1/wihm-variable-friction)

## MuJoCo Simulation Env
A detailed environment description can be found in [env folder](https://github.com/QiyangYan/Variable-Friction-Finger-RL/tree/d7c8c5fd4040c6a2accb320689bcc0b9869805e3/Gymnasium%20Variable%20Friction) **(Incomplete version, comment if you have any question)**, including:
* Action space
* Observation space
* Reward
* Terminated
* Truncated
* Start state and goal state generation

For this section, I will just briefly explain the design of the **potential-based reward function** [1] in addition to the **sparse reward**.
The potential is the negative summation of the radial difference between:
  * The desired radius (`r_left_goal` for the left finger and `r_right_goal` for the right finger) from a contact point on the object to the centre of the corresponding finger motor.
  * The actual achieved radius (`r_left` for the left and `r_right` for the right).

which is further normalised by the finger length `L_finger`, with the formula:
<p align="center">
  <img width="416" alt="Screenshot 2023-09-15 at 12 06 43" src="https://github.com/QiyangYan/Variable-Friction-Finger-RL/assets/75078611/8174c65d-b5af-40f5-9b55-b50ed50e6bb2">
</p>


## Only Sliding Video


https://github.com/QiyangYan/Variable-Friction-Finger-RL/assets/75078611/f1124401-bde0-4848-a911-f7c36ea950ab



## Only Sliding Training Success Rate
Due to computational constraints, I trained only a few epochs to assess the improvement from training. The graph depicts a promising increase in success rate. Interested users are encouraged to extend the training. I will finalize the trainings once I have access to a virtual machine.

<p align="center">
  <img src="https://github.com/QiyangYan/Variable-Friction-Finger-RL/blob/327072ebf893b805a54f24c8bbe9e5ed9a81b635/success_rate.png" alt="Description" width="500">
  <be>
   Success Rate for Sliding
</p>


## Installation and Setup
[Provide steps on how to install and setup the project]



## Usage
[Provide a brief tutorial or steps on how to use the code or run simulations]



## Dependencies
[List out any third-party libraries or dependencies]

## Contributions and Licensing
[Provide details on contributions and any licensing information]

## Reference
[1] https://gibberblot.github.io/rl-notes/single-agent/reward-shaping.html

# Model-Free Gripper Control for Multi-Shape Objects using Reinforcement Learning

## Table of Contents
- [Introduction](#introduction)
- [MuJoCo Simulation](#mujoco-simulation)
- [Only Sliding Training Success Rate](#only-sliding-training-success-rate)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributions and Licensing](#contributions-and-licensing)



## Introduction
Welcome to the repository for our project: Model-Free Gripper Control for Multi-Shape Objects using Reinforcement Learning. This project addresses the limitations of existing algorithms that control in a model-based manner. Leveraging reinforcement learning, our approach achieves multi-shape control without relying on pre-existing models, enhancing the gripper's versatility and adaptability.

The project utilises the [Gymnasium API](https://gymnasium.farama.org) and employs the [MuJoCo](https://mujoco.readthedocs.io/en/stable/overview.html) physics engine for simulations and being trained purely on CPU. Our work also draws inspiration from the [Gymnasium Robotics](https://robotics.farama.org) framework.

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
More detailed environment description could be found in [env folder](https://github.com/QiyangYan/Variable-Friction-Finger-RL/tree/d7c8c5fd4040c6a2accb320689bcc0b9869805e3/Gymnasium%20Variable%20Friction), including:
* Action space
* Observation space
* Reward
* Terminated
* Truncated
* Start state and goal state generation


<p align="center">
  <img src="https://github.com/QiyangYan/Variable-Friction-Finger-RL/assets/75078611/77284c8d-a8d7-46dd-aa4c-ed08e54e5a95" alt="Description" width="400">
  <br>
  MuJoCo Simulation Env
</p>


## Only Sliding Training Result Video


https://github.com/QiyangYan/Variable-Friction-Finger-RL/assets/75078611/f1124401-bde0-4848-a911-f7c36ea950ab



## Only Sliding Training Success Rate
Due to computational constraints, I trained only a few epochs to assess the improvement from training. The graph depicts a promising increase in success rate. Interested users are encouraged to extend the training. I will finalize the trainings once I have access to a virtual machine.

<p align="center">
  <img src="https://github.com/QiyangYan/Variable-Friction-Finger-RL/blob/327072ebf893b805a54f24c8bbe9e5ed9a81b635/success_rate.png" alt="Description" width="500">
  <br>
  Only Sliding Training Success Rate
</p>


## Installation and Setup
[Provide steps on how to install and setup the project]



## Usage
[Provide a brief tutorial or steps on how to use the code or run simulations]



## Dependencies
[List out any third-party libraries or dependencies]

## Contributions and Licensing
[Provide details on contributions and any licensing information]

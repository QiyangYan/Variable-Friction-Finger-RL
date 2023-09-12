# Variable-Friction-Finger-Model-free-RL

## Description


## Action Space
The action space is a `Box(-1, 1, (2,), float32)`. The first control action is the absolute angular positions of the actuated left finger joint. The second control action is the friction state of the gripper. The input of the control actions is set to a range between -1 and 1 by scaling the actual actuator angle ranges. The elements of the action array are the following:

| Num | Action                                              | Control Min | Control Max | Angle Min    | Angle Max   | Name (in corresponding XML file) | Joint | Unit        |
| --- | ----------------------------------------------------| ----------- | ----------- | ------------ | ----------  |--------------------------------- | ----- | ----------- |
| 0   | Angular position of the left finger                 | -1          | 1           | 0 (rad) | 2.0944 (rad)  | robot0:A_WRJ1                    | hinge | angle (rad) |
| 1   | Friction States                                     | -1          | 1           | 0 (rad) | 0.34 (rad) | joint:leftInsert & joint:rightInsert    | hinge | angle (rad) |


## Observation Space
The observation is a goal-aware observation space. It consists of a dictionary with information about the robot’s joint and block states, as well as information about the goal. The dictionary consists of the following 3 keys:

* `observation`: its value is an `ndarray` of shape `(23,)`. It consists of kinematic information of the block object and finger joints. The elements of the array correspond to the following:

| Num | Observation                                                       | Min  | Max  | Joint Name (in corresponding XML file) | Joint Type | Unit                     |
|-----|-------------------------------------------------------------------|------|------|----------------------------------------|------------|--------------------------|
| 0   | Angular position of the right finger joint                       | -Inf | Inf  | robot0:WRJ1                            | hinge      | angle (rad)              |
| 1   | Angular position of the left finger joint                        | -Inf | Inf  | robot0:WRJ0                            | hinge      | angle (rad)              |
| 2   | Angular velocity of the right finger joint                       | -Inf | Inf  | robot0:FFJ3                            | hinge      | angle (rad)              |
| 3   | Angular velocity of the left finger joint                        | -Inf | Inf  | robot0:FFJ2                            | hinge      | angle (rad)              |
| 4   | Angular position of the right finger friction servo              | -Inf | Inf  | robot0:WRJ1                            | hinge      | angle (rad)              |
| 5   | Angular position of the left finger friction servo               | -Inf | Inf  | robot0:WRJ0                            | hinge      | angle (rad)              |
| 6   | Angular velocity of the right finger friction servo              | -Inf | Inf  | robot0:FFJ3                            | hinge      | angle (rad)              |
| 7   | Angular velocity of the left finger friction servo               | -Inf | Inf  | robot0:FFJ2                            | hinge      | angle (rad)              |
| 8   | Linear velocity of the block in x direction                      | -Inf | Inf  | object:joint                           | free       | velocity (m/s)           |
| 9   | Linear velocity of the block in y direction                      | -Inf | Inf  | object:joint                           | free       | velocity (m/s)           |
| 10  | Linear velocity of the block in z direction                      | -Inf | Inf  | object:joint                           | free       | velocity (m/s)           |
| 11  | Angular velocity of the block in x axis                          | -Inf | Inf  | object:joint                           | free       | angular velocity (rad/s) |
| 12  | Angular velocity of the block in y axis                          | -Inf | Inf  | object:joint                           | free       | angular velocity (rad/s) |
| 13  | Angular velocity of the block in z axis                          | -Inf | Inf  | object:joint                           | free       | angular velocity (rad/s) |
| 14  | Position of the block in the x coordinate                        | -Inf | Inf  | object:joint                           | free       | position (m)             |
| 15  | Position of the block in the y coordinate                        | -Inf | Inf  | object:joint                           | free       | position (m)             |
| 16  | Position of the block in the z coordinate                        | -Inf | Inf  | object:joint                           | free       | position (m)             |
| 17  | w component of the quaternion orientation of the block           | -Inf | Inf  | object:joint                           | free       | -                        |
| 18  | x component of the quaternion orientation of the block           | -Inf | Inf  | object:joint                           | free       | -                        |
| 19  | y component of the quaternion orientation of the block           | -Inf | Inf  | object:joint                           | free       | -                        |
| 20  | z component of the quaternion orientation of the block           | -Inf | Inf  | object:joint                           | free       | -                        |
| 21  | Achieved radi between left-contact-point and left motor          | -Inf | Inf  | object:joint                           | free       | -                        |
| 22  | Achieved radi between right-contact-point and right motor        | -Inf | Inf  | object:joint                           | free       | -                        |


* `desired_goal`: this key represents the final goal to be achieved. In this environment, it is a 7-dimensional `ndarray`, `(9,)`, that consists of the pose information of the block. The elements of the array are the following:


| Num | Observation                                                                                                                           | Min    | Max    | Joint Name (in corresponding XML file) | Joint Type | Unit         |
|-----|---------------------------------------------------------------------------------------------------------------------------------------|--------|--------|----------------------------------------|------------|--------------|
| 0   | Target x coordinate of the block                                                                                                      | -Inf   | Inf    | target:joint                           | free       | position (m) |
| 1   | Target y coordinate of the block                                                                                                      | -Inf   | Inf    | target:joint                           | free       | position (m) |
| 2   | Target z coordinate of the block                                                                                                      | -Inf   | Inf    | target:joint                           | free       | position (m) |
| 3   | Target w component of the quaternion orientation of the block                                                                         | -Inf   | Inf    | target:joint                           | free       | -            |
| 4   | Target x component of the quaternion orientation of the block                                                                         | -Inf   | Inf    | target:joint                           | free       | -            |
| 5   | Target y component of the quaternion orientation of the block                                                                         | -Inf   | Inf    | target:joint                           | free       | -            |
| 6   | Target z component of the quaternion orientation of the block                                                                         | -Inf   | Inf    | target:joint                           | free       | -            |
| 7   | Goal radi between left-contact-point and left motor                                                                                   | -Inf   | Inf    | object:joint                           | free       | -            |
| 8   | Goal radi between right-contact-point and right motor                                                                                 | -Inf   | Inf    | object:joint                           | free       | -            |


* `achieved_goal`: this key represents the current state of the block, as if it would have achieved a goal. This is useful for goal-orientated learning algorithms such as those that use **Hindsight Experience Replay (HER)**. The value is an `ndarray` with shape `(9,)`. The elements of the array are the following:


| Num | Observation                                                                                                                           | Min    | Max    | Joint Name (in corresponding XML file) | Joint Type | Unit         |
|-----|---------------------------------------------------------------------------------------------------------------------------------------|--------|--------|----------------------------------------|------------|--------------|
| 0   | Target x coordinate of the block                                                                                                      | -Inf   | Inf    | target:joint                           | free       | position (m) |
| 1   | Target y coordinate of the block                                                                                                      | -Inf   | Inf    | target:joint                           | free       | position (m) |
| 2   | Target z coordinate of the block                                                                                                      | -Inf   | Inf    | target:joint                           | free       | position (m) |
| 3   | Target w component of the quaternion orientation of the block                                                                         | -Inf   | Inf    | target:joint                           | free       | -            |
| 4   | Target x component of the quaternion orientation of the block                                                                         | -Inf   | Inf    | target:joint                           | free       | -            |
| 5   | Target y component of the quaternion orientation of the block                                                                         | -Inf   | Inf    | target:joint                           | free       | -            |
| 6   | Target z component of the quaternion orientation of the block                                                                         | -Inf   | Inf    | target:joint                           | free       | -            |
| 7   | Goal radi between left-contact-point and left motor                                                                                   | -Inf   | Inf    | object:joint                           | free       | -            |
| 8   | Goal radi between right-contact-point and right motor                                                                                 | -Inf   | Inf    | object:joint                           | free       | -            |


## Rewards

The reward can be initialized as `sparse` or `dense`:

* sparse: the returned reward can have two values: `-1` if the block hasn’t reached its final target pose, and `0` if the block is in its final target pose. The block is considered to have reached its final goal if the theta angle difference (theta angle of the 3D axis angle representation is less than 0.1 and if the Euclidean distance to the target position is also less than 0.01 m.
* dense: the returned reward is the negative summation of the Euclidean distance to the block’s target and the theta angle difference to the target orientation. The positional distance is multiplied by a factor of 10 to avoid being dominated by the rotational difference.

To initialize this environment with one of the mentioned reward functions the type of reward must be specified in the id string when the environment is initialized. For `sparse` reward the id is the default of the environment, `VariableFriction-v2`. However, for `dense` reward the id must be modified to `VariableFrictionDense-v2` and initialized as follows:

```python
import gymnasium as gym

env = gym.make('VariableFriction-v2')
```

## Starting State
When the environment is reset the joints of the hand are initialized to their resting position with a 0 displacement. The blocks position and orientation are randomly selected. The initial position is set to (x,y,z)=(, , ) and an offset is added to each coordinate sampled from a normal distribution with 0 mean and 0.005 standard deviation. While the initial orientation is fixed to (w,x,y,z)=(1,0,0,0) to add an angle offset sampled from a uniform distribution with range '[-pi/2, pi/2]'.

The target pose of the block is obtained by adding a random offset to the initial block pose. For the position the offset is sampled from a uniform distribution with range `[(x_min, x_max), (y_min,y_max), (z_min, z_max)] = [(, ), (, ), (, )]`. The orientation offset is sampled from a uniform distribution with range `[-pi/2,pi/2]` and added to one of the Euler axis depending on the environment variation.

## Episode End
The episode will be `truncated` when 
* the duration reaches a total of max_episode_steps which by default is set to 50 timesteps.
* the block moves out of predefined operation range
* the block got stuck during the operation
The episode is never terminated since the task is continuing with infinite horizon.

## Arguments
To increase/decrease the maximum number of timesteps before the episode is truncated the max_episode_steps argument can be set at initialization. The default value is 10. For example, to increase the total number of timesteps to 50 make the environment as follows:

```python
import gymnasium as gym

env = gym.make('HandManipulateBlock-v1', max_episode_steps=50)
```

The same applies for the other environment variations.


## Version History
v2: the environment is designed for complete IHM including rotation and sliding.
v1: the environment is designed under a predefined framework that reduce the exploration range.
v0: the environment is designed only for sliding.

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

* `observation`: its value is an `ndarray` of shape `(22,)`. It consists of kinematic information of the block object and finger joints. The elements of the array correspond to the following:

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

* desired_goal: this key represents the final goal to be achieved. In this environment it is a 7-dimensional ndarray, (7,), that consists of the pose information of the block. The elements of the array are the following:

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


## Rewards

## Starting State

## Episode End

## Arguments

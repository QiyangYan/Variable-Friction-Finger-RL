# Variable-Friction-Finger-Model-free-RL

## Description


## Action Space
The action space is a `Box(-1, 1, (2,), float32)`. The first control action is the absolute angular positions of the actuated left finger joint. The second control action is the friction state of the gripper. The input of the control actions is set to a range between -1 and 1 by scaling the actual actuator angle ranges. The elements of the action array are the following:

| Num | Action                                              | Control Min | Control Max | Angle Min    | Angle Max   | Name (in corresponding XML file) | Joint | Unit        |
| --- | ----------------------------------------------------| ----------- | ----------- | ------------ | ----------  |--------------------------------- | ----- | ----------- |
| 0   | Angular position of the left finger                 | -1          | 1           | -0.489 (rad) | 0.14 (rad)  | robot0:A_WRJ1                    | hinge | angle (rad) |
| 1   | Angular position of the right finger                | -1          | 1           | -0.698 (rad) | 0.489 (rad) | robot0:A_WRJ0                    | hinge | angle (rad) |
| 2   | Friction States                                     | -1          | 1           | -1.571 (rad) | 1.571 (rad) | robot0:A_FFJ3 & robot0:A_FFJ4    | hinge | angle (rad) |


## Observation Space

## Rewards

## Starting State

## Episode End

## Arguments

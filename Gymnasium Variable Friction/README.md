# Variable-Friction-Finger-Model-free-RL

## Description


## Action Space
The action space is a `Box(-1, 1, (2,), float32)`. The first control action is the absolute angular positions of the actuated left finger joint. The second control action is the friction state of the gripper. The input of the control actions is set to a range between -1 and 1 by scaling the actual actuator angle ranges. The elements of the action array are the following:

| Num | Action                                              | Control Min | Control Max | Angle Min    | Angle Max   | Name (in corresponding XML file) | Joint | Unit        |
| --- | ----------------------------------------------------| ----------- | ----------- | ------------ | ----------  |--------------------------------- | ----- | ----------- |
| 0   | Angular position of the left finger                 | -1          | 1           | 0 (rad) | 2.0944 (rad)  | robot0:A_WRJ1                    | hinge | angle (rad) |
| 1   | Friction States                                     | -1          | 1           | 0 (rad) | 0.34 (rad) | joint:leftInsert & joint:rightInsert    | hinge | angle (rad) |


## Observation Space

## Rewards

## Starting State

## Episode End

## Arguments

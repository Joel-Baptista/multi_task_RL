from typing import Any, SupportsFloat, TypeVar

import numpy as np
import copy

from gymnasium import Env, Wrapper
from gymnasium import spaces

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
WrapperActType = TypeVar("WrapperActType")

from enum import Enum

# Finger MCP -> PIP -> DIP
# MCP = Metacarpophalangeal joint
# PIP = Proximal interphalangeal joint
# DIP = Distal interphalangeal joint

JOINTS = [
    "A_ARTx", # Linear translation of the full arm in x direction
    "A_ARTy", # Linear translation of the full arm in y direction
    "A_ARTz", # Linear translation of the full arm in z direction
    "A_ARRx", # Angular up and down movement of the full arm
    "A_ARRy", # Angular left and right movement of the full arm
    "A_ARRz", # Roll angular movement of the full arm
    "A_WRJ1", # Angular position of horizontal wrist joint  (radial/ulnar deviation)
    "A_WRJ0", # Angular position of horizontal wrist joint (flexion/extension)
    "A_FFJ3", # Horizontal angular position of the MCP joint of the forefinger (adduction/abduction)
    "A_FFJ2", # Vertical angular position of the MCP joint of the forefinger (flexion/extension)
    "A_FFJ1", # Angular position of the PIP joint of the forefinger (flexion/extension)
    "A_FFJ0", # Angular position of the DIP joint of the forefinger
    "A_MFJ3", # Horizontal angular position of the MCP joint of the middle finger (adduction/abduction)
    "A_MFJ2", # Vertical angular position of the MCP joint of the middle finger (flexion/extension)
    "A_MFJ1", # Angular position of the PIP joint of the middle finger (flexion/extension)
    "A_MFJ0", # Angular position of the DIP joint of the middle finger
    "A_RFJ3", # Horizontal angular position of the MCP joint of the ring finger (adduction/abduction)
    "A_RFJ2", # Vertical angular position of the MCP joint of the ring finger (flexion/extension)
    "A_RFJ1", # Angular position of the PIP joint of the ring finger (flexion/extension)
    "A_RFJ0", # Angular position of the DIP joint of the ring finger
    "A_LFJ4", # Angular position of the CMC joint of the little finger
    "A_LFJ3", # Horizontal angular position of the MCP joint of the little finger (adduction/abduction)
    "A_LFJ2", # Vertical angular position of the MCP joint of the little finger (flexion/extension)
    "A_LFJ1", # Angular position of the PIP joint of the little finger (flexion/extension)
    "A_LFJ0", # Angular position of the DIP joint of the little finger 
    "A_THJ4", # Horizontal angular position of the MCP joint of the thumb (adduction/abduction)
    "A_THJ3", # Vertical angular position of the MCP joint of the thumb (flexion/extension)
    "A_THJ2", # Angular position of the PIP joint of the thumb (flexion/extension)
    "A_THJ1", # Angular position of the DIP joint of the thumb
    "A_THJ0", # Angular position of the IP joint of the thumb
]

JOINTS_TURNOFF = [
    "A_FFJ3",
    "A_MFJ3",
    "A_RFJ3",
    "A_LFJ4",
    "A_LFJ3",
]

JOINTS_SOLIDARY = { # The joints in the keys are substituted by the joints in the values
    "A_FFJ1": "A_FFJ0",
    "A_MFJ1": "A_MFJ0",
    "A_RFJ1": "A_RFJ0",
    "A_LFJ1": "A_LFJ0",
}

class REDUCE_JOINTS(Wrapper[ObsType, WrapperActType, ObsType, ActType]):
    """Superclass of wrappers that can modify the action before :meth:`env.step`.

    If you would like to apply a function to the action before passing it to the base environment,
    you can simply inherit from :class:`ActionWrapper` and overwrite the method :meth:`action` to implement
    that transformation. The transformation defined in that method must take values in the base environment’s
    action space. However, its domain might differ from the original action space.
    In that case, you need to specify the new action space of the wrapper by setting :attr:`self.action_space` in
    the :meth:`__init__` method of your wrapper.

    Among others, Gymnasium provides the action wrappers :class:`ClipAction` and :class:`RescaleAction` for clipping and rescaling actions.
    """

    def __init__(self, env: Env[ObsType, ActType]):
        """Constructor for the action wrapper."""
        Wrapper.__init__(self, env)

        self.joints = copy.deepcopy(JOINTS)

        for joint_off in JOINTS_TURNOFF:
            self.joints.remove(joint_off)

        for joint_solidary in JOINTS_SOLIDARY:
            self.joints.remove(joint_solidary)

        self.action_space = spaces.Box(-1.0, 1.0, (len(self.joints),), np.float32)

    def step(
        self, action: WrapperActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Runs the :attr:`env` :meth:`env.step` using the modified ``action`` from :meth:`self.action`."""
        return self.env.step(self.action(action))

    def action(self, action: WrapperActType) -> ActType:
        """Returns a modified action before :meth:`env.step` is called.

        Args:
            action: The original :meth:`step` actions

        Returns:
            The modified actions
        """
        full_action = np.zeros(30)


        for i, joint in enumerate(JOINTS):
            if joint in JOINTS_TURNOFF:
                full_action[i] = 0
            elif joint in JOINTS_SOLIDARY.keys():
                full_action[i] = action[self.joints.index(JOINTS_SOLIDARY[joint])]
            else:
                full_action[i] = action[self.joints.index(joint)]

        return full_action
    


#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   :

from thinker_ai.environment.base_env import Environment

# from thinker_ai.environment.android.android_env import AndroidEnv
from thinker_ai.environment.werewolf.werewolf_env import WerewolfEnv
from thinker_ai.environment.stanford_town.stanford_town_env import StanfordTownEnv
from thinker_ai.environment.software.software_env import SoftwareEnv


__all__ = ["AndroidEnv", "WerewolfEnv", "StanfordTownEnv", "SoftwareEnv", "Environment"]

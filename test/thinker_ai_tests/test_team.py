#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : unittest of team

from thinker_ai.agent.roles.project_manager import ProjectManager
from thinker_ai.team import Team


def test_team():
    company = Team()
    company.hire([ProjectManager()])

    assert len(company.env.roles) == 1

from enum import Enum

from thinker_ai.agent.actions.action import Action
from thinker_ai.agent.actions.action_output import ActionOutput
from thinker_ai.agent.actions.add_requirement import UserRequirement
from thinker_ai.agent.actions.debug_error import DebugError
from thinker_ai.agent.actions.design_api import WriteDesign
from thinker_ai.agent.actions.design_api_review import DesignReview
from thinker_ai.agent.actions.project_management import WriteTasks
from thinker_ai.agent.actions.research import CollectLinks, WebBrowseAndSummarize, ConductResearch
from thinker_ai.agent.actions.run_code import RunCode
from thinker_ai.agent.actions.search_and_summarize import SearchAndSummarize
from thinker_ai.agent.actions.write_code import WriteCode
from thinker_ai.agent.actions.write_code_review import WriteCodeReview
from thinker_ai.agent.actions.write_prd import WritePRD
from thinker_ai.agent.actions.write_prd_review import WritePRDReview
from thinker_ai.agent.actions.write_test import WriteTest
from thinker_ai.agent.actions.di.execute_nb_code import ExecuteNbCode
from thinker_ai.agent.actions.di.write_analysis_code import WriteAnalysisCode
from thinker_ai.agent.actions.di.state_machine_task_action import StateMachineTaskPlanAction


class ActionType(Enum):
    """All types of Actions, used for indexing."""

    ADD_REQUIREMENT = UserRequirement
    WRITE_PRD = WritePRD
    WRITE_PRD_REVIEW = WritePRDReview
    WRITE_DESIGN = WriteDesign
    DESIGN_REVIEW = DesignReview
    WRTIE_CODE = WriteCode
    WRITE_CODE_REVIEW = WriteCodeReview
    WRITE_TEST = WriteTest
    RUN_CODE = RunCode
    DEBUG_ERROR = DebugError
    WRITE_TASKS = WriteTasks
    SEARCH_AND_SUMMARIZE = SearchAndSummarize
    COLLECT_LINKS = CollectLinks
    WEB_BROWSE_AND_SUMMARIZE = WebBrowseAndSummarize
    CONDUCT_RESEARCH = ConductResearch
    EXECUTE_NB_CODE = ExecuteNbCode
    WRITE_ANALYSIS_CODE = WriteAnalysisCode
    WRITE_PLAN = StateMachineTaskPlanAction


__all__ = [
    "ActionType",
    "Action",
    "ActionOutput",
]

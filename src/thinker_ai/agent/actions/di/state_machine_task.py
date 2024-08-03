from typing import Type, List, Tuple

from thinker_ai.status_machine.base import Command
from thinker_ai.status_machine.state_machine_definition import StateDefinition, StateMachineDefinitionRepository
from thinker_ai.status_machine.state_machine_context import CompositeStateContext, StateContextBuilder, \
    StateMachineRepository, StateMachineContextBuilder


class StateMachineTask(CompositeStateContext):
    def __init__(self, instance_id: str, instance_group_id: str, state_def: StateDefinition,
                 state_context_builder_class: Type[StateContextBuilder],
                 state_machine_context_repository: StateMachineRepository,
                 state_machine_definition_repository: StateMachineDefinitionRepository):
        super().__init__(instance_id, instance_group_id, state_def, state_context_builder_class,
                         state_machine_context_repository, state_machine_definition_repository)


    def pre_check_plan_from_rsp(self,rsp: str, goal,task_name) -> Tuple[bool, str]:
        try:
            state_machine = (StateMachineContextBuilder
                             .new_from_group_def_json(state_machine_def_group_name=goal,
                                                      state_machine_def_name=task_name,
                                                      def_json=rsp,
                                                      state_machine_definition_repository=self.state_machine_definition_repository,
                                                      state_machine_context_repository=self.state_machine_context_repository)
                             )

            if state_machine:
                results: List[Tuple[List[Command], bool]] = state_machine.self_validate()
                success = True
                fail_paths = []
                for command_list, result in results:
                    if not result:
                        fail_path = []
                        for command in command_list:
                            fail_path.append((command.name, command.target))
                            fail_paths.append(fail_path)
                        success = result
                if success:
                    return True, "状态机验证成功。"
                else:
                    return False, str(fail_paths)
        except Exception as e:
            return False, str(e)


class WritePlan(Action):

    def __init__(self, **data: Any):
        super().__init__(**data)

    async def run(self, goal: str, task_name, instruction: str) -> str:
        if not task_name or not instruction:
            raise Exception("task_name and instruction must be provided")
        create_or_update = "create"
        if not goal:
            goal = task_name
        state_machine_def = definition_repo.get(goal, task_name)
        if state_machine_def:
            create_or_update = "update"

        PROMPT_TEMPLATE: str = """
        #The Name Of State Machine Group:
        {goal}
        #The Name Of State Machine:
        {plan_name}
        # Create Or Update
         you will {create_or_update} this State Machine
        # Instruction:
        {instruction}
        # Context:
        {context}
        # Available Task Types:
        {task_type_desc}
        # Guidance:
        the exist state machine definitions are here,they are related to the current state machine and are the context of the current state machine:
        ```json
         {exist_status_definition}
        ```
        the exist state machine instances are here,they are related to the current state machine and are the context of the current state machine:
        ```json
         {exist_status_instance}
        ```
        {guidance}
        """

        task_type_desc = "\n".join([f"- **{tt.type_name}**: {tt.value.desc}" for tt in TaskType])
        prompt = PROMPT_TEMPLATE.format(
            plan_name=task_name,
            goal=goal,
            create_or_update=create_or_update,
            instruction=instruction,
            context="\n".join([str(ct) for ct in exec_logger.get()]),
            task_type_desc=task_type_desc,
            exist_status_definition=replace_curly_braces(definition_repo.group_to_json(goal)),
            exist_status_instance=replace_curly_braces(instance_repo.group_to_json(goal)),
            guidance=TaskType.STATE_MACHINE_PLAN.value.guidance
        )
        rsp = await self._aask(prompt)
        rsp = CodeParser.parse_code(block=None, text=rsp)
        return rsp
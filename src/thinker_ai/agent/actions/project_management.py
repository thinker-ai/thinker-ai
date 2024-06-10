
import json
from typing import Optional

from thinker_ai.agent.actions.action import Action
from thinker_ai.agent.actions.action_output import ActionOutput
from thinker_ai.agent.provider.schema import Document, Documents
from thinker_ai.agent.actions.project_management_an import PM_NODE, REFINED_PM_NODE
from thinker_ai.common.logs import logger
from thinker_ai.configs.const import PACKAGE_REQUIREMENTS_FILENAME

NEW_REQ_TEMPLATE = """
### Legacy Content
{old_task}

### New Requirements
{context}
"""


class WriteTasks(Action):
    name: str = "CreateTasks"
    i_context: Optional[str] = None

    async def run(self, with_messages):
        changed_system_designs = self.repo.docs.system_design.changed_files
        changed_tasks = self.repo.docs.task.changed_files
        change_files = Documents()
        # Rewrite the system designs that have undergone changes based on the git head diff under
        # `docs/system_designs/`.
        for filename in changed_system_designs:
            task_doc = await self._update_tasks(filename=filename)
            change_files.docs[filename] = task_doc

        # Rewrite the task files that have undergone changes based on the git head diff under `docs/tasks/`.
        for filename in changed_tasks:
            if filename in change_files.docs:
                continue
            task_doc = await self._update_tasks(filename=filename)
            change_files.docs[filename] = task_doc

        if not change_files.docs:
            logger.info("Nothing has changed.")
        # Wait until all files under `docs/tasks/` are processed before sending the publish_message, leaving room for
        # global optimization in subsequent steps.
        return ActionOutput(content=change_files.model_dump_json(), instruct_content=change_files)

    async def _update_tasks(self, filename):
        system_design_doc = await self.repo.docs.system_design.get(filename)
        task_doc = await self.repo.docs.task.get(filename)
        if task_doc:
            task_doc = await self._merge(system_design_doc=system_design_doc, task_doc=task_doc)
            await self.repo.docs.task.save_doc(doc=task_doc, dependencies={system_design_doc.root_relative_path})
        else:
            rsp = await self._run_new_tasks(context=system_design_doc.content)
            task_doc = await self.repo.docs.task.save(
                filename=filename,
                content=rsp.instruct_content.model_dump_json(),
                dependencies={system_design_doc.root_relative_path},
            )
        await self._update_requirements(task_doc)
        return task_doc

    async def _run_new_tasks(self, context):
        node = await PM_NODE.fill(context, self.llm, schema=self.prompt_schema)
        return node

    async def _merge(self, system_design_doc, task_doc) -> Document:
        context = NEW_REQ_TEMPLATE.format(context=system_design_doc.content, old_task=task_doc.content)
        node = await REFINED_PM_NODE.fill(context, self.llm, schema=self.prompt_schema)
        task_doc.content = node.instruct_content.model_dump_json()
        return task_doc

    async def _update_requirements(self, doc):
        m = json.loads(doc.content)
        packages = set(m.get("Required packages", set()))
        requirement_doc = await self.repo.get(filename=PACKAGE_REQUIREMENTS_FILENAME)
        if not requirement_doc:
            requirement_doc = Document(filename=PACKAGE_REQUIREMENTS_FILENAME, root_path=".", content="")
        lines = requirement_doc.content.splitlines()
        for pkg in lines:
            if pkg == "":
                continue
            packages.add(pkg)
        await self.repo.save(filename=PACKAGE_REQUIREMENTS_FILENAME, content="\n".join(packages))

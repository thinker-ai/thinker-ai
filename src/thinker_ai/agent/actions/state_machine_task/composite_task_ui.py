from gradio import gradio as gr, Blocks

from thinker_ai.agent.actions.state_machine_task.composite_task import PlanResult
from thinker_ai.status_machine.state_machine_definition import StateMachineDefinition


class CompositeTaskUI:
    def __init__(self, plan_result: PlanResult):
        self.state_machine_definition: StateMachineDefinition = plan_result.state_machine_definition
        self.result_message: str = plan_result.message
        self.is_success: bool = plan_result.is_success

    def update_state_machine_definition(self, name: str):
        self.state_machine_definition.name = name

    def get_ui(self) -> Blocks:
        with gr.Blocks() as demo:
            gr.Markdown("Start typing below and then click **Run** to see the output.")
            with gr.Row():
                inp = gr.Textbox(placeholder="What is the name?")
                out = gr.Textbox()
            btn = gr.Button("Run")
            btn.click(fn=self.update_state_machine_definition, inputs=inp, outputs=out)
        return demo

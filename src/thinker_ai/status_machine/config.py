from thinker_ai.utils.yaml_model import YamlModel


class StateMachineConfig(YamlModel):
    definition:str = None
    scenario:str = None
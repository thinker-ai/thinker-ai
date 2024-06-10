from thinker_ai.utils.yaml_model import YamlModelWithoutDefault


class S3Config(YamlModelWithoutDefault):
    access_key: str
    secret_key: str
    endpoint: str
    bucket: str

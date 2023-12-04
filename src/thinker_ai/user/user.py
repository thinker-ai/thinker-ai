from abc import ABC


class User(ABC):
    def __init__(self,id:str,name:str):
        self.id = id
        self.name = name
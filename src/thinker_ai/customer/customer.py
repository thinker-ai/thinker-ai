from abc import ABC


class Customer(ABC):
    def __init__(self,id:str,name:str):
        self.id = id
        self.name = name
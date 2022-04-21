from typing import List
from dataclasses import dataclass

Img = str

@dataclass
class Study:
    name: str
    images: List[Img]

    @property
    def number(self):
        return int(self.name.split('study')[-1])

@dataclass
class Patient:
    name: str
    studies: List[Study]

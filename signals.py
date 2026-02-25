from dataclasses import dataclass


@dataclass
class SignalSegment:
    start: int
    "Start index of the segment."
    end: int
    "End index of the segment."

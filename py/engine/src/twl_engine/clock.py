from dataclasses import dataclass


@dataclass(frozen=True)
class Clock:
    """Integer minute clock (MODEL.md section 1); ticks are `tick_minutes` wide."""

    sim_minutes: int
    tick_minutes: int

    def __post_init__(self) -> None:
        if self.tick_minutes <= 0:
            raise ValueError("tick_minutes must be positive")
        if self.sim_minutes < 0 or self.sim_minutes % self.tick_minutes:
            raise ValueError(
                f"sim_minutes {self.sim_minutes} is not a multiple of {self.tick_minutes}"
            )

    @property
    def ticks(self) -> int:
        return self.sim_minutes // self.tick_minutes

    def tick_start(self, tick: int) -> int:
        if not 0 <= tick < self.ticks:
            raise ValueError(f"tick {tick} is outside 0..{self.ticks - 1}")
        return tick * self.tick_minutes

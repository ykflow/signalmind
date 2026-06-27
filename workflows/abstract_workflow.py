from abc import ABC, abstractmethod
from typing import Any, Dict


class AbstractTask(ABC):
    """Abstract base class representing a single atomic operation in a workflow pipeline."""

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the specific isolated task logic.

        Args:
            context: A shared mutable dictionary containing data passed between tasks.

        Returns:
            The updated context dictionary containing the task outputs.
        """
        pass


class AbstractWorkflow(ABC):
    """Abstract base class for orchestrating sequential execution tasks."""

    def __init__(self) -> None:
        self._tasks: list[AbstractTask] = []

    def add_task(self, task: AbstractTask) -> "AbstractWorkflow":
        """Appends an operational task block to the execution chain."""
        if not isinstance(task, AbstractTask):
            raise TypeError("Task must inherit from AbstractTask.")
        self._tasks.append(task)
        return self

    @abstractmethod
    def run(self, initial_context: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the compiled chain of tasks sequentially."""
        pass

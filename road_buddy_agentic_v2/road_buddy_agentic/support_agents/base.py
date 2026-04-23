from __future__ import annotations
from abc import ABC, abstractmethod
from road_buddy_agentic.schemas.support import SupportRequest, SupportBrief


class SupportAgent(ABC):
    def __init__(self, model_name: str = "none", **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    @abstractmethod
    def generate_support_brief(self, request: SupportRequest) -> SupportBrief:
        raise NotImplementedError

"""Inspect the default FinSage knowledge-pipeline routing policy."""

from finsage.configs import RouterCfg, TaskType
from finsage.mind import LLMRouter


def main() -> None:
    """Create a router without making an LLM request."""
    router = LLMRouter(RouterCfg())
    tier = router.cfg.default_routing[TaskType.SUMMARIZATION.value]
    print(f"Summarization is routed to: {tier}")


if __name__ == "__main__":
    main()

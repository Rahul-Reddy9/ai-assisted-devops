# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
"""Agent orchestrator for registering, routing, and coordinating multi-agent workflows."""

from typing import Dict, List, Optional

from devops_ai.agents.base import BaseAgent
from devops_ai.agents.models import (
    AgentCapability,
    AgentContext,
    AgentResult,
    AgentStatus,
    AgentTask,
)
from devops_ai.core.exceptions import AgentNotFoundError
from devops_ai.core.logging import get_logger

logger = get_logger("agents.orchestrator")


class AgentOrchestrator:
    """Coordinates agent registry, task dispatching, and sequential agent pipelines."""

    def __init__(self) -> None:
        self._agents: Dict[str, BaseAgent] = {}
        self.logger = get_logger("agents.orchestrator")

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent instance and verify initialization."""
        if not isinstance(agent, BaseAgent):
            raise TypeError(f"Expected BaseAgent instance, got {type(agent).__name__}")

        # Initialize agent if in IDLE/INITIALIZING
        if agent.status == AgentStatus.IDLE:
            agent.initialize()

        key = agent.name.strip().lower()
        if key in self._agents:
            self.logger.warning(f"Overwriting previously registered agent with name '{agent.name}'")

        self._agents[key] = agent
        self._agents[agent.agent_id.lower()] = agent
        self.logger.info(
            f"Registered agent '{agent.name}' ({agent.agent_id}) with capabilities: "
            f"{[c.value for c in agent.capabilities]}"
        )

    def unregister_agent(self, identifier: str) -> bool:
        """Unregister an agent by name or ID."""
        key = identifier.strip().lower()
        agent = self._agents.get(key)
        if not agent:
            return False

        # Remove both name and agent_id keys
        name_key = agent.name.strip().lower()
        id_key = agent.agent_id.strip().lower()
        self._agents.pop(name_key, None)
        self._agents.pop(id_key, None)
        self.logger.info(f"Unregistered agent '{agent.name}' ({agent.agent_id})")
        return True

    def get_agent(self, identifier: str) -> BaseAgent:
        """Retrieve an agent by name or unique ID."""
        key = identifier.strip().lower()
        agent = self._agents.get(key)
        if not agent:
            raise AgentNotFoundError(identifier)
        return agent

    def list_agents(self) -> List[BaseAgent]:
        """Return list of unique registered agents."""
        seen = set()
        unique_agents: List[BaseAgent] = []
        for agent in self._agents.values():
            if agent.agent_id not in seen:
                seen.add(agent.agent_id)
                unique_agents.append(agent)
        return unique_agents

    def find_agents_by_capability(self, capability: AgentCapability) -> List[BaseAgent]:
        """Filter registered agents by capability."""
        return [a for a in self.list_agents() if capability in a.capabilities]

    def dispatch(
        self,
        task: AgentTask,
        agent_name: Optional[str] = None,
        context: Optional[AgentContext] = None,
    ) -> AgentResult:
        """Route a task to a designated agent or best-match registered agent."""
        if agent_name:
            target_agent = self.get_agent(agent_name)
        else:
            agents = self.list_agents()
            if not agents:
                raise AgentNotFoundError("No agents are currently registered in the orchestrator")
            target_agent = agents[0]

        self.logger.info(
            f"Dispatching task '{task.task_id}' to agent '{target_agent.name}' ({target_agent.agent_id})"
        )
        return target_agent.execute(task, context=context)

    def run_pipeline(
        self,
        task: AgentTask,
        agent_names: List[str],
        initial_context: Optional[AgentContext] = None,
    ) -> List[AgentResult]:
        """Execute a sequential multi-agent pipeline where results and context chain forward."""
        if not agent_names:
            raise ValueError("Pipeline requires at least one agent name")

        ctx = initial_context or AgentContext()
        results: List[AgentResult] = []

        self.logger.info(
            f"Initiating pipeline for task '{task.task_id}' across {len(agent_names)} agents: {agent_names}"
        )

        for name in agent_names:
            agent = self.get_agent(name)
            result = agent.execute(task, context=ctx)
            results.append(result)

            # Accumulate summary into context facts
            ctx.facts.append(f"[{agent.name}] {result.summary}")
            if result.findings:
                ctx.variables[f"{agent.name}_findings"] = result.findings

            # Halt pipeline if an agent fails critically
            if result.status == AgentStatus.FAILED:
                self.logger.warning(
                    f"Pipeline halted early: agent '{agent.name}' failed with error: {result.error}"
                )
                break

        return results

    def broadcast(
        self,
        task: AgentTask,
        agent_names: Optional[List[str]] = None,
        context: Optional[AgentContext] = None,
    ) -> Dict[str, AgentResult]:
        """Broadcast a task concurrently or sequentially to multiple agents."""
        targets = (
            [self.get_agent(name) for name in agent_names] if agent_names else self.list_agents()
        )
        if not targets:
            raise AgentNotFoundError("No target agents available for broadcast")

        results: Dict[str, AgentResult] = {}
        for agent in targets:
            # Each agent receives an independent context view
            agent_ctx = AgentContext(
                facts=list(context.facts) if context else [],
                variables=dict(context.variables) if context else {},
            )
            results[agent.name] = agent.execute(task, context=agent_ctx)

        return results

    def reset_all(self) -> None:
        """Reset all registered agents to IDLE state."""
        for agent in self.list_agents():
            agent.cleanup()

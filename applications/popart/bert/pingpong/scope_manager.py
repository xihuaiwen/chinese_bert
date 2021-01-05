# Copyright 2020 Graphcore Ltd.
"""This module contains utils to setup scope for a Block."""
import inspect
import logging
from contextlib import ExitStack, contextmanager
from typing import Any, Iterator, List, Optional

import popart

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Scope:
    def __init__(self,
                 name: str,
                 vgid: Optional[int] = None,
                 ping_pong_phase: Optional[int] = None,
                 pipeline_stage: Optional[int] = None,
                 additional_scopes: Optional[List]=None):

        if (ping_pong_phase is not None) and (pipeline_stage is not None):
            raise ValueError(
                'Cannot set both `ping_pong_phase` and `pipeline_stage`.')

        if ping_pong_phase is not None:
            self.ping_pong_phase: int = ping_pong_phase
        if pipeline_stage is not None:
            self.pipeline_stage: int = pipeline_stage

        self.name: str = name
        self.vgid: int = vgid
        self.additional_scopes = additional_scopes

    def __setattr__(self, name: str, value: Any):
        if name == 'ping_pong_phase':
            if hasattr(self, 'pipeline_stage'):
                raise ValueError(
                    'Cannot set `ping_pong_phase` when `pipeline_stage` exists.'
                )
        if name == 'pipeline_stage':
            if hasattr(self, 'ping_pong_phase'):
                raise ValueError(
                    'Cannot set `pipeline_stage` when `ping_pong_phase` exists.'
                )
        super(Scope, self).__setattr__(name, value)

    def __repr__(self) -> str:
        if hasattr(self, 'ping_pong_phase'):
            return f'Namescope: {self.name}, Pingpong phase: {self.ping_pong_phase}, VGID: {self.vgid}'
        elif hasattr(self, 'pipeline_stage'):
            return f'Namescope: {self.name}, Pipeline stage: {self.pipeline_stage}, VGID: {self.vgid}'
        if hasattr(self, 'vgid'):
            return f'Namescope: {self.name}, VGID: {self.vgid}'
        return f'Namescope: {self.name}'


class ScopeProvider():
    def __init__(self, start_phase=-1):
        self.prev_phase = start_phase

    def scope_provider(self, builder: popart.Builder,
                       scope: Scope) -> Iterator[ExitStack]:
        """Generate scope for popart layers.

        Args:
            builder (popart.Builder): Builder used to create popart model.
            scope: Scope

        Yields:
            Iterator[ExitStack]: Stack of builder contexts.
        """
        context = ExitStack()
        if hasattr(scope, 'name'):
            context.enter_context(builder.nameScope(scope.name))

        if hasattr(scope, 'ping_pong_phase'):
            if (scope.ping_pong_phase - self.prev_phase) > 1:
                logger.warning('Skipping pingpong scope: {0} -> {1}'.format(
                    self.prev_phase, scope.ping_pong_phase))

            context.enter_context(builder.pingPongPhase(scope.ping_pong_phase))
            self.prev_phase = max(self.prev_phase, scope.ping_pong_phase)

        if hasattr(scope, 'pipeline_stage'):
            context.enter_context(builder.pipelineStage(scope.pipeline_stage))

        if scope.vgid is not None:
            context.enter_context(builder.virtualGraph(scope.vgid))

        if scope.additional_scopes:
            for scope in scope.additional_scopes:
                context.enter_context(scope)

        return context

    @contextmanager
    def __call__(self, builder, scope):
        logger.debug(scope)
        context = self.scope_provider(builder, scope)
        yield context
        context.close()

    def get_next_phase(self) -> int:
        """Get next ping pong phase.

        Returns:
            int: Next ping pong phase.
        """
        self.prev_phase += 1
        return self.prev_phase

    def get_prev_phase(self) -> int:
        """Get last ping pong phase.

        Returns:
            int: Previous ping pong phase.
        """
        return self.prev_phase

    def get_scope(self, name, ping_pong_phase=None, skip_scope=False, additional_scopes=None):
        if inspect.stack()[1].function == 'forward':
            raise ValueError(
                'Scoping must be assigned during layer definition, before the forward pass.'
            )
        if ping_pong_phase is None:
            return Scope(name)

        if ping_pong_phase == 'next':
            if skip_scope:
                self.get_next_phase()
            ping_pong_phase = self.get_next_phase()

        if ping_pong_phase == 'prev':
            ping_pong_phase = self.get_prev_phase()

        vgid = ping_pong_phase % 2

        return Scope(name, vgid, ping_pong_phase, additional_scopes=additional_scopes)

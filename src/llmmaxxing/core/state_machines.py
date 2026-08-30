"""Closed lifecycle/activation state machines shared by Gateway and Control.

Pure transition tables only — persistence, fencing and acknowledgement live in
the gateway/control layers that consume these.
"""

from __future__ import annotations

from typing import final

from llmmaxxing.core.reasons import _Closed


@final
class KeyLifecycleState(_Closed):
    """Client-key lifecycle: DRAFT → ENABLED ↔ SUSPENDED → REVOKED."""

    DRAFT = "draft"
    ENABLED = "enabled"
    SUSPENDED = "suspended"
    REVOKED = "revoked"


@final
class AccountState(_Closed):
    """Provider Account lifecycle; DRAFT means unmeasured/never servable."""

    DRAFT = "draft"
    ACTIVE = "active"
    DISABLED = "disabled"
    TOMBSTONED = "tombstoned"


_KEY_TRANSITIONS: dict[tuple[KeyLifecycleState, str], KeyLifecycleState] = {
    (KeyLifecycleState.DRAFT, "activate"): KeyLifecycleState.ENABLED,
    (KeyLifecycleState.ENABLED, "suspend"): KeyLifecycleState.SUSPENDED,
    (KeyLifecycleState.SUSPENDED, "resume"): KeyLifecycleState.ENABLED,
    # revoke is legal from every state and is terminal (tombstone)
    (KeyLifecycleState.DRAFT, "revoke"): KeyLifecycleState.REVOKED,
    (KeyLifecycleState.ENABLED, "revoke"): KeyLifecycleState.REVOKED,
    (KeyLifecycleState.SUSPENDED, "revoke"): KeyLifecycleState.REVOKED,
    # revoking a tombstone is a no-op, not an error
    (KeyLifecycleState.REVOKED, "revoke"): KeyLifecycleState.REVOKED,
}


def key_transition(state: KeyLifecycleState, event: str) -> KeyLifecycleState:
    """Return the successor state or raise; REVOKED is terminal (no resurrection)."""
    try:
        return _KEY_TRANSITIONS[(state, event)]
    except KeyError:
        raise ValueError(
            f"illegal key lifecycle event {event!r} in state {state.value!r}"
        ) from None


@final
class ActivationStage(_Closed):
    """Immutable-activation pipeline (design: Immutable activation)."""

    PREPARING_BACKEND = "preparing_backend"
    BACKEND_READY = "backend_ready"
    STAGING_GATEWAY = "staging_gateway"
    GATEWAY_STAGED = "gateway_staged"
    COMMITTING = "committing"
    APPLIED = "applied"


_ACTIVATION_ORDER: tuple[ActivationStage, ...] = (
    ActivationStage.PREPARING_BACKEND,
    ActivationStage.BACKEND_READY,
    ActivationStage.STAGING_GATEWAY,
    ActivationStage.GATEWAY_STAGED,
    ActivationStage.COMMITTING,
    ActivationStage.APPLIED,
)


def next_activation_stage(stage: ActivationStage) -> ActivationStage:
    """Advance one strictly linear stage; APPLIED is terminal."""
    index = _ACTIVATION_ORDER.index(stage)
    if index + 1 >= len(_ACTIVATION_ORDER):
        raise ValueError(f"activation stage {stage.value!r} is terminal")
    return _ACTIVATION_ORDER[index + 1]

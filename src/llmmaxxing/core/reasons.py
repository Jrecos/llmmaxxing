"""Closed vocabularies: terminal outcomes, route triggers, strategies, features.

Every enumeration here is *closed*: unknown values fail validation rather than
being ignored or coerced to a default.  New variants require an explicit code
change and (for wire-visible ones) a bundle feature bit.
"""

from __future__ import annotations

import enum
from typing import final


class _Closed(enum.StrEnum):
    """Base that rejects unknown spellings with a plain ValueError."""


@final
class TerminalOutcome(_Closed):
    """Closed set of terminal request outcomes (design: Deterministic routing).

    ``QUEUED`` is deliberately absent: it is nonterminal and therefore not a
    legal value here.
    """

    AUTHZ_DENIED = "authz_denied"
    AUTH_STATE_UNAVAILABLE = "auth_state_unavailable"
    UNSUPPORTED_REQUEST = "unsupported_request"
    BACKPRESSURE_REJECTED = "backpressure_rejected"
    ROUTE_UNAVAILABLE = "route_unavailable"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    UPSTREAM_FAILED = "upstream_failed"
    CLIENT_CANCELLED = "client_cancelled"
    RESPONSE_STREAM_FAILED = "response_stream_failed"
    COMPLETED = "completed"


@final
class RouteTrigger(_Closed):
    """Typed causes that may activate a non-primary leg (design: V1 triggers)."""

    PRIMARY = "primary"
    CAPACITY_SPILL = "capacity_spill"
    FAILURE_FALLBACK = "failure_fallback"
    QUOTA_FALLBACK = "quota_fallback"
    MANUAL_EMERGENCY = "manual_emergency"
    SHADOW = "shadow"


@final
class RouteStrategy(_Closed):
    """V1 supports exactly one deterministic strategy; others fail closed."""

    ORDERED_CAPACITY = "ordered_capacity"


@final
class EndpointKind(_Closed):
    """Certified V1 inference endpoints; unsupported paths never coerce."""

    CHAT = "chat"
    TEXT = "text"
    EMBEDDINGS = "embeddings"
    RERANK = "rerank"
    AUDIO_SPEECH = "audio_speech"
    AUDIO_TRANSCRIPTION = "audio_transcription"
    IMAGE = "image"


@final
class DispatchCause(_Closed):
    """One serving cause selected before a provider attempt."""

    PRIMARY = "primary"
    CAPACITY = "capacity"
    FAILURE = "failure"
    QUOTA = "quota"
    MANUAL_EMERGENCY = "manual_emergency"


@final
class FailureCause(_Closed):
    """Result of the single provider-failure classifier."""

    CAPACITY = "capacity"
    TRANSIENT_FAILURE = "transient_failure"
    QUOTA = "quota"
    UNKNOWN = "unknown"


@final
class FailureScope(_Closed):
    """Circuit boundary proven by normalized provider evidence."""

    ACCOUNT = "account"
    DEPLOYMENT = "deployment"
    UNKNOWN = "unknown"


@final
class Modality(_Closed):
    """Request modality accepted by the V1 profiler."""

    CHAT = "chat"
    TEXT = "text"
    RESPONSES = "responses"
    MESSAGES = "messages"
    EMBEDDING = "embedding"
    RERANK = "rerank"
    AUDIO_SPEECH = "audio_speech"
    AUDIO_TRANSCRIPTION = "audio_transcription"
    AUDIO_TRANSLATION = "audio_translation"
    IMAGE = "image"


@final
class QuotaDimensionStatus(_Closed):
    """Whether one provider quota dimension is bounded, unknown, or unlimited."""

    KNOWN = "known"
    UNKNOWN = "unknown"
    ATTESTED_ABSENT = "attested_absent"


@final
class RequiredFeature(_Closed):
    """Closed V1 bundle feature bits."""

    ORDERED_CAPACITY = "ordered_capacity"
    WEIGHTED_FAIR_QUEUE = "weighted_fair_queue"
    EXPIRY_DENY_OVERLAY = "expiry_deny_overlay"
    EMERGENCY_LEG_ACTIVATION = "emergency_leg_activation"
    SHADOW_LEGS = "shadow_legs"
    CREDENTIAL_GENERATION_FENCE = "credential_generation_fence"


#: Feature bits a bundle may require.  A Gateway binary must understand every
#: listed feature before applying the bundle; unknown names are rejected at
#: model validation time, never silently skipped.
V1_FEATURES: frozenset[str] = frozenset(feature.value for feature in RequiredFeature)

#: Highest Gateway reader version this build can satisfy.  A bundle whose
#: ``min_reader`` exceeds it cannot be applied by V1 binaries (downgrade guard).
MAX_MIN_READER: tuple[int, int] = (1, 0)

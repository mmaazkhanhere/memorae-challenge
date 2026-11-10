"""Minimal, untrusted Vapi server-message contracts."""

from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class VapiCaller(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    phone_number: str | None = Field(
        default=None,
        validation_alias=AliasChoices("phoneNumber", "number"),
    )


class VapiCall(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str = Field(min_length=1, max_length=255)
    caller: VapiCaller | None = Field(default=None, alias="from")
    customer: VapiCaller | None = None

    @property
    def caller_phone_number(self) -> str | None:
        """Return Vapi's inbound caller ID without exposing its source downstream.

        Vapi documents ``call.from.phoneNumber`` for inbound server events.
        Some provider call records instead supply the equivalent caller as
        ``call.customer.number``. Both values remain untrusted evidence until
        they pass the existing normalized linked-identity lookup.
        """

        return (
            self.caller.phone_number
            if self.caller is not None and self.caller.phone_number
            else self.customer.phone_number
            if self.customer is not None
            else None
        )

    @field_validator("id")
    @classmethod
    def _strip_id(cls, value: str) -> str:
        candidate = value.strip()
        if not candidate:
            raise ValueError("call id must not be blank")
        return candidate


class VapiMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str = Field(min_length=1, max_length=100)
    call: VapiCall | None = None
    tool_call_list: list[VapiToolCall] = Field(
        default_factory=list,
        validation_alias=AliasChoices("toolCallList", "tool_call_list"),
    )

    @field_validator("type")
    @classmethod
    def _strip_type(cls, value: str) -> str:
        candidate = value.strip()
        if not candidate:
            raise ValueError("message type must not be blank")
        return candidate


class VapiServerRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message: VapiMessage


class VapiFunctionCall(BaseModel):
    """Untrusted function-call data supplied by Vapi."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    name: str = Field(min_length=1, max_length=100)
    arguments: Any = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        candidate = value.strip()
        if not candidate:
            raise ValueError("function name must not be blank")
        return candidate


class VapiToolCall(BaseModel):
    """One Vapi function invocation awaiting an application result."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str = Field(min_length=1, max_length=255)
    type: str = Field(default="function", min_length=1, max_length=50)
    function: VapiFunctionCall

    @field_validator("id")
    @classmethod
    def _strip_id(cls, value: str) -> str:
        candidate = value.strip()
        if not candidate:
            raise ValueError("tool call id must not be blank")
        return candidate


__all__ = [
    "VapiCall",
    "VapiCaller",
    "VapiFunctionCall",
    "VapiMessage",
    "VapiServerRequest",
    "VapiToolCall",
]

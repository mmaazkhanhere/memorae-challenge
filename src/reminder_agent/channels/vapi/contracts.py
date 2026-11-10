"""Minimal, untrusted Vapi server-message contracts."""

from __future__ import annotations

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


__all__ = ["VapiCall", "VapiCaller", "VapiMessage", "VapiServerRequest"]

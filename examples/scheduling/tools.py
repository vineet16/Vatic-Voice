"""Tool definitions for the clinic scheduling agent."""

from __future__ import annotations

from pydantic import BaseModel, Field

from examples.scheduling.backend import BackendPool, SlotTaken
from vatic.core.tools import SideEffect, ToolContext, ToolError, ToolRegistry, ToolSpec


class LookupIn(BaseModel):
    name: str = Field(description="The patient's full name (first and last).")


class Appointment(BaseModel):
    id: str
    date: str
    time: str
    provider: str


class Patient(BaseModel):
    id: str
    first_name: str
    last_name: str
    next_appointment: Appointment | None = None


class LookupOut(BaseModel):
    found: bool
    patient: Patient | None = None


class AvailabilityIn(BaseModel):
    date: str = Field(description="ISO date, YYYY-MM-DD.")


class AvailabilityOut(BaseModel):
    date: str
    available: bool
    times: list[str]


class BookIn(BaseModel):
    patient_id: str
    date: str = Field(description="ISO date, YYYY-MM-DD.")
    time: str = Field(description="24h time, HH:MM.")


class BookOut(BaseModel):
    appointment_id: str
    date: str
    time: str
    provider: str


class RescheduleIn(BaseModel):
    appointment_id: str
    date: str = Field(description="ISO date, YYYY-MM-DD.")
    time: str = Field(description="24h time, HH:MM.")


class CancelIn(BaseModel):
    appointment_id: str


class CancelOut(BaseModel):
    cancelled: bool
    appointment_id: str


def build_registry(pool: BackendPool) -> ToolRegistry:
    def lookup_patient(args: LookupIn, ctx: ToolContext) -> LookupOut:
        return LookupOut.model_validate(pool.get(ctx.session_id).lookup_patient(args.name))

    def check_availability(args: AvailabilityIn, ctx: ToolContext) -> AvailabilityOut:
        return AvailabilityOut.model_validate(
            pool.get(ctx.session_id).check_availability(args.date)
        )

    def book_appointment(args: BookIn, ctx: ToolContext) -> BookOut:
        try:
            out = pool.get(ctx.session_id).book_appointment(args.patient_id, args.date, args.time)
        except SlotTaken as exc:
            raise ToolError(str(exc)) from exc
        return BookOut.model_validate(out)

    def reschedule_appointment(args: RescheduleIn, ctx: ToolContext) -> BookOut:
        try:
            out = pool.get(ctx.session_id).reschedule_appointment(
                args.appointment_id, args.date, args.time
            )
        except SlotTaken as exc:
            raise ToolError(str(exc)) from exc
        return BookOut.model_validate(out)

    def cancel_appointment(args: CancelIn, ctx: ToolContext) -> CancelOut:
        try:
            out = pool.get(ctx.session_id).cancel_appointment(args.appointment_id)
        except SlotTaken as exc:
            raise ToolError(str(exc)) from exc
        return CancelOut.model_validate(out)

    return ToolRegistry(
        [
            ToolSpec(
                name="lookup_patient",
                description="Find a patient by full name; includes their next appointment.",
                input_schema=LookupIn,
                output_schema=LookupOut,
                side_effect=SideEffect.READ_ONLY,
                handler=lookup_patient,
            ),
            ToolSpec(
                name="check_availability",
                description="List open appointment times on a date.",
                input_schema=AvailabilityIn,
                output_schema=AvailabilityOut,
                side_effect=SideEffect.READ_ONLY,
                handler=check_availability,
            ),
            ToolSpec(
                name="book_appointment",
                description="Book an appointment. Confirm with the caller first.",
                input_schema=BookIn,
                output_schema=BookOut,
                side_effect=SideEffect.IRREVERSIBLE,
                invariants=["args.date != '' and args.time != ''"],
                handler=book_appointment,
            ),
            ToolSpec(
                name="reschedule_appointment",
                description="Move an existing appointment to a new date and time.",
                input_schema=RescheduleIn,
                output_schema=BookOut,
                side_effect=SideEffect.REVERSIBLE,
                handler=reschedule_appointment,
            ),
            ToolSpec(
                name="cancel_appointment",
                description="Cancel an appointment. Confirm with the caller first.",
                input_schema=CancelIn,
                output_schema=CancelOut,
                side_effect=SideEffect.IRREVERSIBLE,
                handler=cancel_appointment,
            ),
        ]
    )

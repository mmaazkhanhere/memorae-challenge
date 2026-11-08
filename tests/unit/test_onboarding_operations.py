from sqlalchemy import select

from reminder_agent.features.onboarding.identity_operations import (
    prepare_phone_link,
    resolve_linked_identity,
)
from reminder_agent.features.onboarding.phone_linking import (
    process_phone_link as process_feature_phone_link,
)
from reminder_agent.features.onboarding.phone_linking_service import process_phone_link
from reminder_agent.infrastructure.db.models import ChannelIdentity, User


def test_prepare_phone_link_stages_local_number_until_region_is_known(db):
    user = User(name="Ada", onboarding_status="ACTIVE", timezone="UTC")
    db.add(user)
    db.commit()

    result = prepare_phone_link(
        db,
        user=user,
        phone_number="03001234567",
        default_region=None,
    )
    db.flush()

    pending = db.scalar(
        select(ChannelIdentity).where(
            ChannelIdentity.user_id == user.id,
            ChannelIdentity.channel_type == "PHONE",
            ChannelIdentity.status == "PENDING",
        )
    )
    assert result.outcome == "MISSING_REGION"
    assert result.reason == "missing_region"
    assert pending is not None
    assert pending.external_id == "03001234567"


def test_prepare_phone_link_returns_normalized_value_without_linking(db):
    user = User(name="Ada", onboarding_status="ACTIVE", timezone="UTC")
    db.add(user)
    db.commit()

    result = prepare_phone_link(
        db,
        user=user,
        phone_number="+44 20 8366 1177",
        default_region=None,
    )

    assert result.outcome == "NORMALIZED"
    assert result.normalized == "+442083661177"
    assert db.scalar(select(ChannelIdentity).where(ChannelIdentity.user_id == user.id)) is None


def test_resolve_linked_identity_is_channel_neutral(db):
    user = User(name="Ada", onboarding_status="PENDING")
    db.add(user)
    db.flush()
    identity = ChannelIdentity(
        user_id=user.id,
        channel_type="PHONE",
        external_id="+442083661177",
        status="LINKED",
    )
    db.add(identity)
    db.commit()

    resolved = resolve_linked_identity(
        db,
        channel_type="PHONE",
        external_id="+442083661177",
    )

    assert resolved is not None
    assert resolved.user.id == user.id
    assert resolved.identity.id == identity.id
    assert (
        resolve_linked_identity(
            db,
            channel_type="PHONE",
            external_id="+14155550123",
        )
        is None
    )


def test_phone_link_workflow_returns_channel_neutral_decisions(db):
    user = User(name="Ada", onboarding_status="ACTIVE", timezone="UTC")
    db.add(user)
    db.commit()

    missing_region = process_phone_link(
        db,
        user=user,
        phone_number="03001234567",
        default_region=None,
    )
    assert missing_region.outcome == "MISSING_REGION"
    assert missing_region.reason == "missing_region"

    linked = process_phone_link(
        db,
        user=user,
        phone_number="+44 20 8366 1177",
        default_region=None,
    )
    assert linked.outcome == "LINKED"
    assert linked.normalized == "+442083661177"


def test_feature_phone_link_decision_is_usable_with_fake_channel_ports():
    class Preparation:
        outcome = "NORMALIZED"
        normalized = "+442083661177"
        reason = None
        region = None

    class Mutation:
        outcome = "LINKED"
        reason = None
        replaced_existing = False

    result = process_feature_phone_link(
        object(),
        user=object(),
        phone_number="ignored by fake port",
        default_region=None,
        prepare_phone_link_fn=lambda *_args, **_kwargs: Preparation(),
        link_phone_identity_fn=lambda *_args, **_kwargs: Mutation(),
    )

    assert result.outcome == "LINKED"
    assert result.normalized == "+442083661177"


def test_feature_phone_link_decision_preserves_channel_neutral_failures():
    class MissingRegion:
        outcome = "MISSING_REGION"
        normalized = None
        reason = "missing_region"
        region = None

    missing_region = process_feature_phone_link(
        object(),
        user=object(),
        phone_number="03001234567",
        default_region=None,
        prepare_phone_link_fn=lambda *_args, **_kwargs: MissingRegion(),
        link_phone_identity_fn=lambda *_args, **_kwargs: None,
    )

    class ConflictPreparation:
        outcome = "NORMALIZED"
        normalized = "+442083661177"
        reason = None
        region = "GB"

    class ConflictMutation:
        outcome = "CONFLICT"
        reason = "active_number_conflict"
        replaced_existing = False

    conflict = process_feature_phone_link(
        object(),
        user=object(),
        phone_number="ignored by fake port",
        default_region="GB",
        prepare_phone_link_fn=lambda *_args, **_kwargs: ConflictPreparation(),
        link_phone_identity_fn=lambda *_args, **_kwargs: ConflictMutation(),
    )

    assert missing_region.outcome == "MISSING_REGION"
    assert missing_region.reason == "missing_region"
    assert conflict.outcome == "CONFLICT"
    assert conflict.reason == "active_number_conflict"

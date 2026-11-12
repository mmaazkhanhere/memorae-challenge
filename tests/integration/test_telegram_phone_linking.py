import os
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from reminder_agent.channels.telegram.provider import TelegramUpdate
from reminder_agent.features.onboarding.message_templates import ONBOARDING_MESSAGES
from reminder_agent.features.onboarding.workflow import handle_update
from reminder_agent.infrastructure.database.models import ChannelIdentity, IdempotencyRecord, User


def post_update(client, update: dict):
    return client.post(
        "/webhook/telegram",
        json=update,
        headers={"X-Telegram-Bot-Api-Secret-Token": "test-secret"},
    )


def onboard_user(
    api_client,
    update_factory,
    *,
    user_id: int = 1001,
    chat_id: int = 2001,
    start_update_id: int = 100,
) -> None:
    assert (
        post_update(
            api_client,
            update_factory(start_update_id, user_id=user_id, chat_id=chat_id, text="/start"),
        ).status_code
        == 200
    )
    assert (
        post_update(
            api_client,
            update_factory(
                start_update_id + 1,
                user_id=user_id,
                chat_id=chat_id,
                text="Asia/Karachi",
            ),
        ).status_code
        == 200
    )


def contact_update(
    update_factory,
    update_id: int,
    *,
    user_id: int = 1001,
    chat_id: int = 2001,
    phone_number: str = "+442083661177",
    contact_user_id: int | None = 1001,
) -> dict:
    contact = {"phone_number": phone_number}
    if contact_user_id is not None:
        contact["user_id"] = contact_user_id
    return update_factory(
        update_id,
        user_id=user_id,
        chat_id=chat_id,
        text=None,
        contact=contact,
    )


def test_timezone_completion_offers_native_contact_keyboard(
    api_client, telegram_client, update_factory
) -> None:
    onboard_user(api_client, update_factory)

    markup = telegram_client.reply_markups[-1]
    assert markup is not None
    assert markup["keyboard"][0][0]["request_contact"] is True
    assert markup["keyboard"][1][0]["text"] == "Skip for now"
    assert telegram_client.messages[-1][1] in {
        template.format(name="Ada", timezone_name="Asia/Karachi")
        for template in ONBOARDING_MESSAGES["onboarding_complete"]
    }


def test_phone_region_without_country_code_explains_required_format(
    api_client, telegram_client, update_factory
) -> None:
    onboard_user(api_client, update_factory)

    response = post_update(api_client, update_factory(102, text="/phone_region"))

    assert response.status_code == 200
    assert "/phone_region PK" in telegram_client.messages[-1][1]
    assert telegram_client.messages[-1][1] in set(ONBOARDING_MESSAGES["phone_region_missing"])


def test_valid_self_contact_links_normalized_phone(
    api_client, telegram_client, db: Session, update_factory
) -> None:
    onboard_user(api_client, update_factory)
    response = post_update(api_client, contact_update(update_factory, 102))

    assert response.status_code == 200
    assert telegram_client.messages[-1][1] in set(ONBOARDING_MESSAGES["phone_linked"])
    identity = db.scalar(select(ChannelIdentity).where(ChannelIdentity.channel_type == "PHONE"))
    assert identity is not None
    assert identity.external_id == "+442083661177"
    assert identity.status == "LINKED"
    assert identity.verified_at is not None
    assert telegram_client.reply_markups[-1] == {"remove_keyboard": True}


def test_contact_from_another_user_is_rejected_without_persistence(
    api_client, telegram_client, db: Session, update_factory
) -> None:
    onboard_user(api_client, update_factory)
    response = post_update(
        api_client,
        contact_update(update_factory, 102, contact_user_id=9999),
    )

    assert response.status_code == 200
    assert telegram_client.messages[-1][1] in set(ONBOARDING_MESSAGES["phone_ownership_rejected"])
    assert (
        db.scalar(
            select(func.count())
            .select_from(ChannelIdentity)
            .where(ChannelIdentity.channel_type == "PHONE")
        )
        == 0
    )


def test_contact_without_owner_id_is_rejected(
    api_client, telegram_client, db: Session, update_factory
) -> None:
    onboard_user(api_client, update_factory)
    response = post_update(
        api_client,
        contact_update(update_factory, 102, contact_user_id=None),
    )

    assert response.status_code == 200
    assert telegram_client.messages[-1][1] in set(ONBOARDING_MESSAGES["phone_ownership_rejected"])
    assert (
        db.scalar(
            select(func.count())
            .select_from(ChannelIdentity)
            .where(ChannelIdentity.channel_type == "PHONE")
        )
        == 0
    )


def test_invalid_phone_is_rejected_without_persistence(
    api_client, telegram_client, db: Session, update_factory
) -> None:
    onboard_user(api_client, update_factory)
    response = post_update(
        api_client,
        contact_update(update_factory, 102, phone_number="not-a-phone"),
    )

    assert response.status_code == 200
    assert telegram_client.messages[-1][1] in set(ONBOARDING_MESSAGES["phone_number_invalid_retry"])
    assert (
        db.scalar(
            select(func.count())
            .select_from(ChannelIdentity)
            .where(ChannelIdentity.channel_type == "PHONE")
        )
        == 0
    )


def test_local_contact_can_use_user_selected_region(
    api_client, telegram_client, db: Session, update_factory
) -> None:
    onboard_user(api_client, update_factory)
    response = post_update(
        api_client,
        contact_update(update_factory, 102, phone_number="03001234567"),
    )

    assert response.status_code == 200
    assert telegram_client.messages[-1][1] in set(ONBOARDING_MESSAGES["local_phone_region_required"])
    assert (
        db.scalar(
            select(func.count())
            .select_from(ChannelIdentity)
            .where(
                ChannelIdentity.channel_type == "PHONE",
                ChannelIdentity.status == "PENDING",
            )
        )
        == 1
    )

    assert post_update(api_client, update_factory(103, text="/phone_region PK")).status_code == 200
    assert telegram_client.messages[-1][1] in set(ONBOARDING_MESSAGES["phone_linked"])

    identity = db.scalar(
        select(ChannelIdentity).where(
            ChannelIdentity.channel_type == "PHONE",
            ChannelIdentity.status == "LINKED",
        )
    )
    assert identity is not None
    assert identity.external_id == "+923001234567"
    assert db.scalar(select(User).where(User.phone_region == "PK")) is not None


def test_phone_cannot_be_linked_to_two_active_users(
    api_client, telegram_client, db: Session, update_factory
) -> None:
    onboard_user(api_client, update_factory, user_id=1001, chat_id=2001)
    assert post_update(api_client, contact_update(update_factory, 102)).status_code == 200

    onboard_user(api_client, update_factory, user_id=2002, chat_id=3002, start_update_id=200)
    response = post_update(
        api_client,
        contact_update(
            update_factory,
            202,
            user_id=2002,
            chat_id=3002,
            contact_user_id=2002,
        ),
    )

    assert response.status_code == 200
    assert telegram_client.messages[-1][1] in set(ONBOARDING_MESSAGES["phone_conflict"])
    assert (
        db.scalar(
            select(func.count())
            .select_from(ChannelIdentity)
            .where(ChannelIdentity.channel_type == "PHONE")
        )
        == 1
    )


def test_active_phone_uniqueness_is_enforced_by_database(
    api_client, db: Session, update_factory
) -> None:
    onboard_user(api_client, update_factory)
    first_user = db.scalar(select(User))
    assert first_user is not None
    second_user = User(name="Bea", timezone="Asia/Karachi", onboarding_status="ACTIVE")
    db.add(second_user)
    db.commit()

    db.add(
        ChannelIdentity(
            user_id=first_user.id,
            channel_type="PHONE",
            external_id="+442083661177",
            status="LINKED",
        )
    )
    db.commit()
    db.add(
        ChannelIdentity(
            user_id=second_user.id,
            channel_type="PHONE",
            external_id="+442083661177",
            status="LINKED",
        )
    )
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


def test_replayed_contact_update_is_idempotent(
    api_client, telegram_client, db: Session, update_factory
) -> None:
    onboard_user(api_client, update_factory)
    update = contact_update(update_factory, 102)
    assert post_update(api_client, update).status_code == 200
    message_count = len(telegram_client.messages)
    assert post_update(api_client, update).status_code == 200

    assert len(telegram_client.messages) == message_count
    assert (
        db.scalar(
            select(func.count())
            .select_from(ChannelIdentity)
            .where(ChannelIdentity.channel_type == "PHONE")
        )
        == 1
    )
    assert db.scalar(select(func.count()).select_from(IdempotencyRecord)) == 3


def test_skip_keeps_telegram_account_usable_without_phone(
    api_client, telegram_client, db: Session, update_factory
) -> None:
    onboard_user(api_client, update_factory)
    response = post_update(
        api_client,
        update_factory(102, text="Skip for now"),
    )

    assert response.status_code == 200
    assert telegram_client.messages[-1][1] in set(ONBOARDING_MESSAGES["phone_link_skipped"])
    assert (
        db.scalar(
            select(func.count())
            .select_from(ChannelIdentity)
            .where(ChannelIdentity.channel_type == "PHONE")
        )
        == 0
    )

    assert post_update(api_client, update_factory(103, text="/link_phone")).status_code == 200
    assert telegram_client.messages[-1][1] in set(ONBOARDING_MESSAGES["phone_link_prompt"])


def test_relinking_replaces_the_users_previous_phone(
    api_client, db: Session, update_factory
) -> None:
    onboard_user(api_client, update_factory)
    assert post_update(api_client, contact_update(update_factory, 102)).status_code == 200
    assert (
        post_update(
            api_client,
            contact_update(update_factory, 103, phone_number="+923001234567"),
        ).status_code
        == 200
    )

    identities = db.scalars(
        select(ChannelIdentity).where(ChannelIdentity.channel_type == "PHONE")
    ).all()
    assert sorted((identity.external_id, identity.status) for identity in identities) == [
        ("+442083661177", "UNLINKED"),
        ("+923001234567", "LINKED"),
    ]


@pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL") or os.getenv("TEST_DATABASE_URL", "").startswith("sqlite"),
    reason="requires a real PostgreSQL database for concurrent constraint coverage",
)
def test_concurrent_phone_links_allow_only_one_owner(
    api_client, db: Session, session_factory, update_factory
) -> None:
    onboard_user(api_client, update_factory, user_id=1001, chat_id=2001, start_update_id=300)
    onboard_user(api_client, update_factory, user_id=2002, chat_id=3002, start_update_id=400)

    def link_phone(user_id: int, chat_id: int, update_id: int) -> str:
        session = session_factory()
        try:
            result = handle_update(
                session,
                TelegramUpdate.model_validate(
                    contact_update(
                        update_factory,
                        update_id,
                        user_id=user_id,
                        chat_id=chat_id,
                        contact_user_id=user_id,
                    )
                ),
            )
            assert result is not None
            return result.text
        finally:
            session.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda values: link_phone(*values),
                [(1001, 2001, 302), (2002, 3002, 402)],
            )
        )

    assert sum(result in ONBOARDING_MESSAGES["phone_linked"] for result in results) == 1
    assert sum(result in ONBOARDING_MESSAGES["phone_conflict"] for result in results) == 1
    assert (
        db.scalar(
            select(func.count())
            .select_from(ChannelIdentity)
            .where(
                ChannelIdentity.channel_type == "PHONE",
                ChannelIdentity.status == "LINKED",
            )
        )
        == 1
    )

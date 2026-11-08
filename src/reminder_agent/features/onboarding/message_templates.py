"""Randomized user-facing Telegram onboarding messages."""

from __future__ import annotations

import random

ONBOARDING_MESSAGES: dict[str, list[str]] = {
    "welcome": [
        "Hey {name}! 👋 I'm here to help keep the little things off your mind. First things first, what city or time zone are you in? (e.g. Asia/Karachi)",
        "Hi {name}! Nice to meet you. I'll help you keep track of things so you don't have to remember everything yourself. What city or time zone should I use? (e.g. Europe/Madrid)",
        "Hey {name}, welcome aboard! 👋 Think of me as your extra memory. Before we get started, what city or time zone should I use (e.g. America/Chicago)?",
        "Hi {name}! 👋 Got a lot to remember? That's what I'm here for. Let's start with your city or time zone.(e.g. Europe/London)",
    ],
    "welcome_back": [
        "Hey {name}, welcome back! 👋 We were just getting you set up. What city or time zone should I use? (e.g. Europe/London)",
        "Good to see you again, {name}! Just one thing left before we get started . Can you share your time zone. (e.g. America/Chicago)",
        "Hey {name}! Let's pick up where we left off. What city or time zone are you in? (e.g. Asia/Karachi)",
    ],
    "already_onboarded": [
        "Hey {name}! 👋 Good to have you back. What's on your mind?",
        "You're back, {name}! Anything you want me to remember?",
        "Hey {name}! I'm here. What are we keeping track of today?",
        "Welcome back, {name}! Got something you don't want to forget?",
    ],
    "already_onboarded_without_phone": [
        "Hey {name}, you're all set on Telegram! By the way, you can connect your phone whenever you'd like to talk to me on a call.",
        "Welcome back, {name}! Everything's ready here. If you ever want to reach me by phone, we can set that up too.",
        "You're good to go, {name}! Telegram works as usual, and phone calls are just an optional step away.",
    ],
    "unknown_sender": [
        "Hey there! 👋 We haven't met yet. Send /start and we'll get you set up.",
        "Hi! 👋 Send /start and I'll help you get started.",
        "Looks like we're new to each other! Send /start to begin.",
    ],
    "requires_timezone_for_phone": [
        "Almost there! I just need your time zone before we connect your phone.",
        "First, let's set your time zone. Then we can connect your phone.",
        "One quick thing before phone setup  what time zone are you in?",
    ],
    "phone_already_linked": [
        "You're already connected! No need to share your number again.",
        "Your phone's already linked. You're good to go! 📞",
        "All set! That phone number is already connected to your account.",
    ],
    "phone_link_prompt": [
        "Want to take me with you on calls too? 📞 Share your own phone number using the button below and I'll recognize you when you call. Or we can leave that for later.",
        "One more thing, if you'd like! You can talk to me over the phone too. Just share your number using the button below, or skip for now.",
        "Prefer talking sometimes? 📞 Link your number and we can pick up on a phone call with the same reminders and memories. Totally optional!",
    ],
    "phone_link_skipped": [
        "Of course! We can stick to Telegram for now. You can connect your phone whenever you feel like it.",
        "Sounds good! No phone needed. I'm right here whenever you've got something to remember.",
        "No worries! Telegram is ready to go. We can always connect your phone later.",
    ],
    "phone_remains_linked": [
        "Nothing's changed  your phone is still connected.",
        "You're still connected! Your phone number remains linked.",
        "All good! Your phone connection is exactly as you left it.",
    ],
    "phone_region_missing": [
        "Just need your country code first! Try /phone_region PK for Pakistan.",
        "I need your phone's country code. Send something like /phone_region PK.",
    ],
    "phone_region_invalid": [
        "That country code doesn't look quite right. Try a two-letter code like PK, GB, or US.",
        "Hmm, I couldn't recognize that country code. Could you try PK, GB, or US?",
    ],
    "phone_region_saved": [
        "Got it, {region}! Now tap Share my phone number and we'll finish connecting.",
        "Perfect! I've saved {region} as your phone country. Go ahead and share your number using the button.",
        "Country set to {region}! Just share your phone number and we're nearly done.",
    ],
    "phone_number_invalid": [
        "Hmm, that number doesn't seem to match {region}. Could you check the country code or try sharing it again?",
        "I couldn't verify that number for {region}. Mind checking the country code and trying again?",
    ],
    "phone_link_unknown_sender": [
        "We haven't set things up yet! Send /start first, and then we can connect your phone.",
        "Hey there! Let's get you started before linking a phone. Just send /start.",
    ],
    "phone_link_requires_onboarding": [
        "Just one thing before we connect your phone  let's finish setting up your time zone.",
        "We're almost ready! Share your time zone first, then we'll get your phone connected.",
    ],
    "phone_ownership_rejected": [
        "For your security, I can only connect your own number. Please use Telegram's Share my phone number button.",
        "I can only link a number that belongs to you. Please share your own number using Telegram's button.",
    ],
    "local_phone_region_required": [
        "Looks like Telegram sent your number without a country code. Send something like /phone_region PK, then share it again.",
        "I just need your phone's country code to check the number. Try /phone_region PK, then share your number again.",
    ],
    "phone_number_invalid_retry": [
        "I couldn't verify that one. Mind trying to share your number again?",
        "That number didn't quite work. Please try sharing it again using the button.",
    ],
    "phone_conflict": [
        "That number is already connected to another account, so I can't link it here. Please try a different one.",
        "Looks like that phone number is already in use on another account. Could you try a different number?",
    ],
    "phone_linked": [
        "And we're connected! 📞 You can now reach me by calling from this number.",
        "Nice! Your phone's linked. Whether you're messaging here or calling, I'll know it's you.",
        "All set! 🎉 Your phone is connected. You're ready to talk whenever you need me.",
    ],
    "invalid_timezone": [
        "Hmm, I'm not finding that time zone. Could you try something like Asia/Karachi?",
        "That time zone doesn't look quite right. Try an identifier like Asia/Karachi or Europe/London.",
        "I couldn't recognize that one. Mind trying a time zone like Asia/Karachi?",
    ],
    "onboarding_complete": [
        "Perfect, {name}! 🎉 I've got your time zone as {timezone_name}. Now, what's the first thing you'd like me to remember? You can also share your phone number below to enable calls.",
        "We're all set, {name}! I'll use {timezone_name} for your reminders. Want to try one? Say 'Remind me to stretch in 5 minutes.' You can also share your phone number below for phone calls.",
        "And that's it, {name}! You're ready to go. I'll keep your reminders on {timezone_name} time. What's on your mind? If you'd like calls too, share your phone number below.",
        "Lovely, {name}! Your time zone is set to {timezone_name}. You can start telling me what you'd like to remember, and share your phone number below if you'd like phone calls.",
    ],
}


def render_message(key: str, **values: str) -> str:
    """Randomly select and interpolate one message for an operation."""

    return random.choice(ONBOARDING_MESSAGES[key]).format(**values)


def welcome(name: str) -> str:
    return render_message("welcome", name=name)


def welcome_back(name: str) -> str:
    return render_message("welcome_back", name=name)


def already_onboarded(name: str) -> str:
    return render_message("already_onboarded", name=name)


def already_onboarded_without_phone(name: str) -> str:
    return render_message("already_onboarded_without_phone", name=name)


def unknown_sender() -> str:
    return render_message("unknown_sender")


def requires_timezone_for_phone() -> str:
    return render_message("requires_timezone_for_phone")


def phone_already_linked() -> str:
    return render_message("phone_already_linked")


def phone_link_prompt() -> str:
    return render_message("phone_link_prompt")


def phone_link_skipped() -> str:
    return render_message("phone_link_skipped")


def phone_remains_linked() -> str:
    return render_message("phone_remains_linked")


def phone_region_missing() -> str:
    return render_message("phone_region_missing")


def phone_region_invalid() -> str:
    return render_message("phone_region_invalid")


def phone_region_saved(region: str) -> str:
    return render_message("phone_region_saved", region=region)


def phone_number_invalid(region: str) -> str:
    return render_message("phone_number_invalid", region=region)


def phone_link_unknown_sender() -> str:
    return render_message("phone_link_unknown_sender")


def phone_link_requires_onboarding() -> str:
    return render_message("phone_link_requires_onboarding")


def phone_ownership_rejected() -> str:
    return render_message("phone_ownership_rejected")


def local_phone_region_required() -> str:
    return render_message("local_phone_region_required")


def phone_number_invalid_retry() -> str:
    return render_message("phone_number_invalid_retry")


def phone_conflict() -> str:
    return render_message("phone_conflict")


def phone_linked() -> str:
    return render_message("phone_linked")


def invalid_timezone() -> str:
    return render_message("invalid_timezone")


def onboarding_complete(name: str, timezone_name: str) -> str:
    return render_message("onboarding_complete", name=name, timezone_name=timezone_name)


__all__ = [
    "ONBOARDING_MESSAGES",
    "already_onboarded",
    "already_onboarded_without_phone",
    "invalid_timezone",
    "local_phone_region_required",
    "onboarding_complete",
    "phone_already_linked",
    "phone_conflict",
    "phone_link_prompt",
    "phone_link_skipped",
    "phone_link_unknown_sender",
    "phone_link_requires_onboarding",
    "phone_linked",
    "phone_number_invalid",
    "phone_number_invalid_retry",
    "phone_ownership_rejected",
    "phone_region_invalid",
    "phone_region_missing",
    "phone_region_saved",
    "phone_remains_linked",
    "render_message",
    "requires_timezone_for_phone",
    "unknown_sender",
    "welcome",
    "welcome_back",
]

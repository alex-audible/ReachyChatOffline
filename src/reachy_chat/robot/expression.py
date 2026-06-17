"""Lightweight reply -> expression mapper for Reachy Mini.

The voice pipeline does **not** use structured LLM function-calling, so the robot
cannot be told "play the happy move" by the model directly. Instead, this module
maps an assistant reply's *content* to one of the recorded emotion moves so the
robot reacts expressively to its own spoken words. The mapping is a deterministic,
fast keyword/sentiment heuristic -- no ML, no network, no per-token cost. It is
safe to call synchronously right after a reply is produced; the chosen move name
is then handed to the decoupled motion layer (see ``reply_to_move`` /
``Expression`` and the usage note at the bottom of this docstring).

It is grounded in the **actual** 81 moves of the live
``pollen-robotics/reachy-mini-emotions-library`` dataset (confirmed against the
running daemon, not guessed). Move names referenced here all exist in that set::

    amazed1 anxiety1 attentive1 attentive2 boredom1 boredom2 calming1 cheerful1
    come1 confused1 contempt1 curious1 dance1 dance2 dance3 disgusted1
    displeased1 displeased2 downcast1 dying1 electric1 enthusiastic1
    enthusiastic2 exhausted1 fear1 frustrated1 furious1 go_away1 grateful1
    helpful1 helpful2 impatient1 impatient2 incomprehensible2 indifferent1
    inquiring1 inquiring2 inquiring3 irritated1 irritated2 laughing1 laughing2
    lonely1 lost1 loving1 no1 no_excited1 no_sad1 oops1 oops2 proud1 proud2
    proud3 rage1 relief1 relief2 reprimand1 reprimand2 reprimand3 resigned1
    sad1 sad2 scared1 serenity1 shy1 sleep1 success1 success2 surprised1
    surprised2 thoughtful1 thoughtful2 tired1 uncertain1 uncomfortable1
    understanding1 understanding2 welcoming1 welcoming2 yes1 yes_sad1

Usage (after each assistant reply, off the speech latency path)::

    from reachy_chat.robot.expression import reply_to_move

    move = reply_to_move(reply_text)            # -> "welcoming2" | None
    if move:
        queue.enqueue(PlayMoveCommand(move_name=move), interrupt=True)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# --------------------------------------------------------------------------- #
# The recorded emotion moves we map onto. This is the *intent vocabulary* of the
# heuristic; every name here exists in the live emotions-library dataset. The
# mapper never invents a move name -- if no rule fires it returns None (neutral).
# --------------------------------------------------------------------------- #
EMOTION_LIBRARY: tuple[str, ...] = (
    "amazed1", "anxiety1", "attentive1", "attentive2", "boredom1", "boredom2",
    "calming1", "cheerful1", "come1", "confused1", "contempt1", "curious1",
    "dance1", "dance2", "dance3", "disgusted1", "displeased1", "displeased2",
    "downcast1", "dying1", "electric1", "enthusiastic1", "enthusiastic2",
    "exhausted1", "fear1", "frustrated1", "furious1", "go_away1", "grateful1",
    "helpful1", "helpful2", "impatient1", "impatient2", "incomprehensible2",
    "indifferent1", "inquiring1", "inquiring2", "inquiring3", "irritated1",
    "irritated2", "laughing1", "laughing2", "lonely1", "lost1", "loving1",
    "no1", "no_excited1", "no_sad1", "oops1", "oops2", "proud1", "proud2",
    "proud3", "rage1", "relief1", "relief2", "reprimand1", "reprimand2",
    "reprimand3", "resigned1", "sad1", "sad2", "scared1", "serenity1", "shy1",
    "sleep1", "success1", "success2", "surprised1", "surprised2", "thoughtful1",
    "thoughtful2", "tired1", "uncertain1", "uncomfortable1", "understanding1",
    "understanding2", "welcoming1", "welcoming2", "yes1", "yes_sad1",
)
_LIBRARY_SET = frozenset(EMOTION_LIBRARY)


@dataclass(frozen=True)
class Expression:
    """The outcome of mapping a reply to a recorded move.

    Attributes:
        move: The chosen recorded-move name (always in :data:`EMOTION_LIBRARY`),
            or ``None`` for a neutral reply that warrants no emotion move.
        category: The intent bucket that fired (e.g. ``"greeting"``,
            ``"neutral"``) -- useful for logging / tuning.
        rationale: A short human-readable reason, e.g. the trigger that matched.
    """

    move: Optional[str]
    category: str
    rationale: str


# --------------------------------------------------------------------------- #
# Heuristic rules. Each rule is (category, move, regex-of-triggers). The first
# rule whose pattern matches wins, so rules are ordered most-specific-first.
# Patterns match on word boundaries over the lowercased reply, so "hi" does not
# fire inside "this" and "no" does not fire inside "now".
#
# `move` is the recorded move to play; every one is in EMOTION_LIBRARY. Choosing
# a single deterministic move per category keeps the mapping reproducible (no
# RNG) and easy to reason about / test.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Rule:
    category: str
    move: Optional[str]
    triggers: tuple[str, ...]


# Triggers are matched as whole words/phrases (see _compile). Apostrophes in
# contractions ("i'm", "that's") are handled by the tokeniser below.
_RULES: tuple[_Rule, ...] = (
    # --- Farewell: leaving the conversation. Checked early; "bye"/"goodbye"
    #     should win over an incidental greeting word.
    _Rule(
        "farewell",
        "go_away1",
        (
            "goodbye", "bye", "bye bye", "farewell", "see you", "see ya",
            "take care", "talk to you later", "catch you later", "good night",
            "goodnight", "have a good", "have a great day", "until next time",
            "so long",
        ),
    ),
    # --- Greeting: opening the conversation -> a welcoming wave.
    _Rule(
        "greeting",
        "welcoming2",
        (
            "hello", "hi", "hey", "hi there", "good morning", "good afternoon",
            "good evening", "greetings", "welcome", "nice to meet you",
            "pleased to meet you", "how can i help", "how may i help",
            "what can i do for you",
        ),
    ),
    # --- Apology / mistake -> a small "oops". Checked before generic sadness so
    #     "sorry, I made a mistake" reads as apologetic rather than sad.
    _Rule(
        "apology",
        "oops1",
        (
            "sorry", "i apologise", "i apologize", "my apologies", "my bad",
            "oops", "i made a mistake", "my mistake", "i was wrong",
            "i got that wrong", "forgive me",
        ),
    ),
    # --- Gratitude -> a grateful gesture.
    _Rule(
        "grateful",
        "grateful1",
        (
            "thank you", "thanks", "thank you so much", "i appreciate it",
            "i appreciate that", "much appreciated", "grateful",
        ),
    ),
    # --- Sadness / bad news -> a downcast move.
    _Rule(
        "sad",
        "sad1",
        (
            "i'm sad", "im sad", "that's sad", "thats sad", "so sad",
            "unfortunately", "i'm afraid not", "im afraid not",
            "bad news", "that's a shame", "thats a shame", "what a pity",
            "i can't help", "i cant help", "i'm unable", "im unable",
            "regrettably", "heartbroken",
        ),
    ),
    # --- Excitement / happiness -> an enthusiastic, upbeat move. Broad: covers
    #     exclamation-flavoured positivity.
    _Rule(
        "excited",
        "enthusiastic1",
        (
            "amazing", "awesome", "fantastic", "wonderful", "excellent",
            "great news", "i'm so excited", "im so excited", "so exciting",
            "exciting", "can't wait", "cant wait", "love it", "i love",
            "brilliant", "incredible", "yay", "hooray", "woohoo", "let's go",
            "lets go", "absolutely", "thrilled", "delighted", "perfect",
        ),
    ),
    # --- Plain positive / done -> a cheerful, satisfied move.
    _Rule(
        "happy",
        "cheerful1",
        (
            "happy", "glad", "good", "great", "nice", "wonderful", "pleased",
            "all done", "all set", "there you go", "here you go", "sounds good",
            "no problem", "you're welcome", "youre welcome", "of course",
        ),
    ),
    # --- Success / accomplishment -> a proud "success" move.
    _Rule(
        "success",
        "success1",
        (
            "done", "finished", "completed", "success", "succeeded", "it worked",
            "that worked", "fixed", "solved", "got it", "ready",
        ),
    ),
    # --- Agreement / affirmation -> a nod-flavoured "yes".
    _Rule(
        "agreement",
        "yes1",
        (
            "yes", "yeah", "yep", "yup", "sure", "okay", "ok", "alright",
            "agreed", "i agree", "correct", "exactly", "indeed", "definitely",
            "that's right", "thats right", "you're right", "youre right",
            "will do", "sounds right",
        ),
    ),
    # --- Surprise -> an amazed move. Checked before negation so "no way"
    #     reads as surprise rather than a bare "no" refusal.
    _Rule(
        "surprised",
        "amazed1",
        (
            "wow", "whoa", "woah", "surprising", "surprised", "unbelievable",
            "no way", "really?", "oh my", "that's surprising", "thats surprising",
        ),
    ),
    # --- Negation / refusal -> a "no" gesture.
    _Rule(
        "negation",
        "no1",
        (
            "no", "nope", "not really", "i don't think so", "i dont think so",
            "that's not", "thats not", "that's incorrect", "thats incorrect",
            "i disagree", "absolutely not", "i can't do that", "i cant do that",
        ),
    ),
    # --- Curiosity / question back -> a curious head-tilt move. Broad keyword
    #     set; the trailing "?" is also caught as a fallback below.
    _Rule(
        "curious",
        "curious1",
        (
            "what do you", "what would you", "what about you", "how about you",
            "tell me more", "i'm curious", "im curious", "i wonder",
            "do you", "would you like", "can you tell me", "what's your",
            "whats your", "which one", "anything else",
        ),
    ),
    # --- Thinking / uncertainty -> a thoughtful move.
    _Rule(
        "thinking",
        "thoughtful1",
        (
            "let me think", "let me see", "hmm", "i'm not sure", "im not sure",
            "not entirely sure", "it depends", "perhaps", "maybe", "possibly",
            "i think", "let me check", "good question",
        ),
    ),
    # --- Confusion -> a confused move.
    _Rule(
        "confused",
        "confused1",
        (
            "i don't understand", "i dont understand", "i do not understand",
            "i'm confused", "im confused", "that doesn't make sense",
            "that doesnt make sense",
            "what do you mean", "i'm not following", "im not following",
            "could you clarify", "can you clarify", "i'm lost", "im lost",
        ),
    ),
)


def _normalise(text: str) -> str:
    """Lowercase and collapse whitespace; keep ``?`` and apostrophes."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def _contains_phrase(haystack: str, phrase: str) -> bool:
    """Whole-word/phrase containment so 'hi' != 'this' and 'no' != 'now'.

    A token may legitimately end in ``?`` (e.g. "really?"), so the right
    boundary allows an optional question mark.
    """
    # Word boundary on the left; on the right allow end, whitespace, or common
    # trailing punctuation (so "yes." / "yes!" / "yes," all match "yes").
    pattern = r"(?<![\w'])" + re.escape(phrase) + r"(?![\w'])"
    return re.search(pattern, haystack) is not None


def map_reply(text: Optional[str]) -> Expression:
    """Map an assistant reply to an :class:`Expression` (move + rationale).

    The first matching rule wins (rules are ordered most-specific-first). If no
    keyword rule fires, a trailing question mark falls back to *curious*, and an
    exclamation mark falls back to *cheerful*; otherwise the reply is neutral and
    no move is played.

    Args:
        text: The assistant reply content. ``None``/empty -> neutral.

    Returns:
        An :class:`Expression`. ``expression.move`` is ``None`` for neutral
        replies, otherwise a recorded-move name guaranteed to be in
        :data:`EMOTION_LIBRARY`.
    """
    if not text or not text.strip():
        return Expression(None, "neutral", "empty reply")

    norm = _normalise(text)

    for rule in _RULES:
        for trigger in rule.triggers:
            if _contains_phrase(norm, trigger):
                return Expression(
                    rule.move,
                    rule.category,
                    f"matched '{trigger}' -> {rule.category}",
                )

    # Punctuation fallbacks: a question reads as curiosity, an exclamation as
    # cheer. These only fire when no keyword rule matched.
    stripped = norm.rstrip(" \"')")
    if stripped.endswith("?"):
        return Expression("curious1", "curious", "ends with '?' -> curious")
    if stripped.endswith("!"):
        return Expression("cheerful1", "happy", "ends with '!' -> cheerful")

    return Expression(None, "neutral", "no sentiment cue matched")


def reply_to_move(text: Optional[str]) -> Optional[str]:
    """Return the recorded-move name for a reply, or ``None`` if neutral.

    Thin convenience wrapper over :func:`map_reply` for callers that only want
    the move name. Deterministic and fast (pure string matching).

    Example::

        move = reply_to_move("Hello there! How can I help?")  # -> "welcoming2"
    """
    return map_reply(text).move

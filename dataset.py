"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
    # expanded
    "joy",
    "joyful",
    "glad",
    "thrilled",
    "grateful",
    "thankful",
    "proud",
    "hopeful",
    "optimistic",
    "wonderful",
    "fantastic",
    "brilliant",
    "delighted",
    "cheerful",
    "pumped",
    "stoked",
    "content",
    "peaceful",
    "calm",
    "energized",
    "motivated",
    "inspired",
    "confident",
    "blessed",
    "nice",
    "fine",
    "okay",
    "better",
    "refreshed",
    "thriving",
    "winning",
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
    # expanded
    "cry",
    "crying",
    "miserable",
    "depressed",
    "hopeless",
    "lonely",
    "scared",
    "anxious",
    "worried",
    "nervous",
    "frustrated",
    "annoyed",
    "disappointed",
    "hurt",
    "pain",
    "suffering",
    "exhausted",
    "drained",
    "overwhelmed",
    "lost",
    "broken",
    "numb",
    "dread",
    "dreading",
    "regret",
    "guilty",
    "ashamed",
    "embarrassed",
    "pathetic",
    "terrible",
    "horrible",
    "disgusting",
    "sick",
    "struggling",
]

# ---------------------------------------------------------------------
# Starter labeled dataset
# ---------------------------------------------------------------------

# Short example posts written as if they were social media updates or messages.
SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
]

# Human labels for each post above.
# Allowed labels in the starter:
#   - "positive"
#   - "negative"
#   - "neutral"
#   - "mixed"
TRUE_LABELS = [
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",     # "Feeling tired but kind of hopeful"
    "neutral",   # "This is fine"
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this"
]

# TODO: Add 5-10 more posts and labels.
#
SAMPLE_POSTS.append("Lowkey stressed but kind of proud of myself")
TRUE_LABELS.append("mixed")

SAMPLE_POSTS.append("I absolutely love getting stuck in traffic :)")
TRUE_LABELS.append("negative")  # sarcasm

SAMPLE_POSTS.append("No cap, this project is actually amazing 💀")
TRUE_LABELS.append("positive")


SAMPLE_POSTS.append("Highkey obsessed but also terrified")
TRUE_LABELS.append("mixed")

SAMPLE_POSTS.append("Just vibing, nothing really matters anyway :(")
TRUE_LABELS.append("neutral")  # ambiguous apathy

SAMPLE_POSTS.append("This is the best worst day ever")
TRUE_LABELS.append("mixed")

SAMPLE_POSTS.append("Honestly just exhausted and over it")
TRUE_LABELS.append("negative")

SAMPLE_POSTS.append("Love going to work every day :)")
TRUE_LABELS.append("negative")  # sarcasm

# Additional examples to improve ML generalization
SAMPLE_POSTS.append("I feel like crying today")
TRUE_LABELS.append("negative")

SAMPLE_POSTS.append("I am so excited about the presentation")
TRUE_LABELS.append("positive")

SAMPLE_POSTS.append("Feeling sad now but I guess I will be fine")
TRUE_LABELS.append("mixed")

SAMPLE_POSTS.append("I can not stop crying, everything is wrong")
TRUE_LABELS.append("negative")

SAMPLE_POSTS.append("Today was amazing, I am so happy")
TRUE_LABELS.append("positive")

SAMPLE_POSTS.append("I am really anxious about tomorrow")
TRUE_LABELS.append("negative")

SAMPLE_POSTS.append("Just got some great news, feeling blessed")
TRUE_LABELS.append("positive")

SAMPLE_POSTS.append("Nothing feels right lately")
TRUE_LABELS.append("negative")

SAMPLE_POSTS.append("So grateful for everything in my life")
TRUE_LABELS.append("positive")

SAMPLE_POSTS.append("I am exhausted and overwhelmed")
TRUE_LABELS.append("negative")

SAMPLE_POSTS.append("Feeling pumped and motivated today")
TRUE_LABELS.append("positive")

SAMPLE_POSTS.append("Kind of nervous but also excited")
TRUE_LABELS.append("mixed")

SAMPLE_POSTS.append("Everything is going wrong and I feel lost")
TRUE_LABELS.append("negative")

SAMPLE_POSTS.append("Had a wonderful day with my friends")
TRUE_LABELS.append("positive")

SAMPLE_POSTS.append("I regret so many things")
TRUE_LABELS.append("negative")

SAMPLE_POSTS.append("Feeling content and at peace")
TRUE_LABELS.append("positive")

SAMPLE_POSTS.append("Worried about the future but staying hopeful")
TRUE_LABELS.append("mixed")

SAMPLE_POSTS.append("I hate how things turned out")
TRUE_LABELS.append("negative")

SAMPLE_POSTS.append("Thrilled about what is coming next")
TRUE_LABELS.append("positive")

SAMPLE_POSTS.append("Feeling okay I guess, not great not bad")
TRUE_LABELS.append("neutral")
#
# Tips:
#   - Try to create some examples that are hard to label even for you.
#   - Make a note of any examples that you and a friend might disagree on.
#     Those "edge cases" are interesting to inspect for both the rule based
#     and ML models.
#
# Example of how you might extend the lists:
#
# SAMPLE_POSTS.append("Lowkey stressed but kind of proud of myself")
# TRUE_LABELS.append("mixed")
#
# Remember to keep them aligned:
#   len(SAMPLE_POSTS) == len(TRUE_LABELS)

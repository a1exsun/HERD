"""USPTO-50K Draft-Verify defaults and shared label constants."""

DRAFT_K: int = 5
VERIFY_CONFIRM_K: int = 5
VERIFY_CHALLENGE_K: int = 5
COLD_START_THRESHOLD: int = 10

# TRACE+17 labels are the class names themselves (none is numeric), so no number → name map.
CLASS_NAMES: dict[int, str] = {}

_OPTIONS_STR = "\n".join(f"  {k}. {v}" for k, v in CLASS_NAMES.items())

# Canonical user-facing instruction prefix shared by data prep + harness prompts.
INSTRUCTION: str = (
    "You are an expert organic chemist. Given the following reaction SMILES, "
    "classify the reaction into one of the 10 reaction types listed below. "
    "Output only the class number inside [class] and <eoa> tags. "
    "For example: [class]3<eoa>\n\n"
    "Reaction types:\n"
    + _OPTIONS_STR
)

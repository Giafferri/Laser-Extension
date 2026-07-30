"""Define and validate the fixed FEVER prompt-template suite."""

from collections import OrderedDict
import re


CLAIM_SLOT = "{claim}"

_CATEGORY_SPECS = OrderedDict(
    [
        ("lex", "Lexical substitutions"),
        ("syn", "Syntactic rephrasings"),
        ("inst", "Instruction framing"),
        ("slot", "Answer-slot framing"),
        ("verb", "Verbosity / explicitness"),
    ]
)


def _normalize_claim(claim):
    claim = claim.strip()
    if claim.endswith((".", "?")):
        return claim
    return claim + "."


def _collapse_whitespace(text):
    return " ".join(text.split())


def _wrapper_without_claim(template):
    return _collapse_whitespace(template.replace(CLAIM_SLOT, ""))


def _word_tokens(text):
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text.lower())


def _levenshtein_distance(left, right):
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    prev = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        curr = [i]
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def _jaccard_distance(left_tokens, right_tokens):
    left_set = set(left_tokens)
    right_set = set(right_tokens)
    union = left_set | right_set
    if not union:
        return 0.0
    return 1.0 - (len(left_set & right_set) / len(union))


def _prompt_record(template_id, category_id, category_label, variant_index, template, is_original=False):
    return {
        "template_id": template_id,
        "category_id": category_id,
        "category_label": category_label,
        "variant_index": variant_index,
        "template": template,
        "is_original": is_original,
    }


def _build_prompt_records():
    records = [
        _prompt_record(
            template_id="original",
            category_id="original",
            category_label="Original",
            variant_index=0,
            template="Consider the following claim: {claim} Is this claim true or false. The claim is",
            is_original=True,
        )
    ]

    lex_pairs = [
        ("Read", "claim"),
        ("Review", "claim"),
        ("Examine", "claim"),
        ("Inspect", "claim"),
        ("Read", "statement"),
        ("Read", "assertion"),
        ("Read", "proposition"),
        ("Read", "sentence"),
    ]
    for index, (verb, noun) in enumerate(lex_pairs, start=1):
        records.append(
            _prompt_record(
                template_id=f"lex_{index:02d}",
                category_id="lex",
                category_label=_CATEGORY_SPECS["lex"],
                variant_index=index,
                template=f"{verb} the following {noun} and decide whether it is true or false: {{claim}} The {noun} is",
            )
        )

    syn_leads = [
        "Consider whether the following claim is true or false:",
        "For the claim below, decide whether it is true or false:",
        "Decide whether the claim below is true or false:",
        "Determine whether the following claim is true or false:",
        "Judge whether the claim below is true or false:",
        "Is the following claim true or false?",
        "True or false: consider the following claim.",
        "The following claim may be true or false. Decide which:",
    ]
    for index, lead in enumerate(syn_leads, start=1):
        records.append(
            _prompt_record(
                template_id=f"syn_{index:02d}",
                category_id="syn",
                category_label=_CATEGORY_SPECS["syn"],
                variant_index=index,
                template=f"{lead} {{claim}} The claim is",
            )
        )

    inst_leads = [
        "Act as a fact checker and determine whether the following claim is true or false:",
        "As a verifier, decide whether the following claim is true or false:",
        "Your task is to judge whether the following claim is true or false:",
        "Please decide whether the following claim is true or false:",
        "Respond by deciding whether the following claim is true or false:",
        "Evaluate the factual status of the following claim as true or false:",
        "Classify the following claim as true or false:",
        "Provide a true-or-false judgment for the following claim:",
    ]
    for index, lead in enumerate(inst_leads, start=1):
        records.append(
            _prompt_record(
                template_id=f"inst_{index:02d}",
                category_id="inst",
                category_label=_CATEGORY_SPECS["inst"],
                variant_index=index,
                template=f"{lead} {{claim}} The claim is",
            )
        )

    slot_intro = "Consider the following claim: {claim} Decide whether it is true or false."
    slot_suffixes = [
        "The claim is",
        "This claim is",
        "The statement is",
        "This statement is",
        "The assertion is",
        "The correct label is",
        "The correct answer is",
        "It is",
    ]
    for index, suffix in enumerate(slot_suffixes, start=1):
        records.append(
            _prompt_record(
                template_id=f"slot_{index:02d}",
                category_id="slot",
                category_label=_CATEGORY_SPECS["slot"],
                variant_index=index,
                template=f"{slot_intro} {suffix}",
            )
        )

    verb_templates = [
        "Claim: {claim} True or false? The claim is",
        "Consider the following claim: {claim} Determine its truth value, choosing either true or false. The claim is",
        "Consider the following claim carefully: {claim} Decide whether the claim is true or false. The claim is",
        "Consider the following factual claim: {claim} Decide whether this claim is true or false. The claim is",
        "Read the claim below carefully. Determine whether it should be labeled true or false: {claim} The claim is",
        "You will be given a claim. Decide whether the claim is true or false and respond accordingly: {claim} The claim is",
        "Below is a claim that must be evaluated for truthfulness. Decide whether it is true or false: {claim} The claim is",
        "Evaluate the factual accuracy of the following claim. Use a true-or-false judgment: {claim} The claim is",
    ]
    for index, template in enumerate(verb_templates, start=1):
        records.append(
            _prompt_record(
                template_id=f"verb_{index:02d}",
                category_id="verb",
                category_label=_CATEGORY_SPECS["verb"],
                variant_index=index,
                template=template,
            )
        )

    return records


def _augment_prompt_records(records):
    original_template = next(record["template"] for record in records if record["template_id"] == "original")
    original_wrapper = _wrapper_without_claim(original_template)
    original_tokens = _word_tokens(original_wrapper)

    augmented = []
    for record in records:
        spec = dict(record)
        wrapper = _wrapper_without_claim(spec["template"])
        wrapper_tokens = _word_tokens(wrapper)
        max_len = max(len(wrapper), len(original_wrapper))
        spec["wrapper"] = wrapper
        spec["wrapper_char_len"] = len(wrapper)
        spec["wrapper_word_len"] = len(wrapper_tokens)
        spec["divergence_char_norm"] = (
            0.0 if max_len == 0 else _levenshtein_distance(wrapper, original_wrapper) / max_len
        )
        spec["divergence_token_jaccard"] = _jaccard_distance(wrapper_tokens, original_tokens)
        augmented.append(spec)
    return augmented


FEVER_PROMPT_SPECS = tuple(_augment_prompt_records(_build_prompt_records()))
FEVER_PROMPT_SPECS_BY_ID = OrderedDict((spec["template_id"], spec) for spec in FEVER_PROMPT_SPECS)
FEVER_PROMPT_TEMPLATES = OrderedDict((spec["template_id"], spec["template"]) for spec in FEVER_PROMPT_SPECS)


def lint_fever_prompt_spec(spec):
    issues = []
    template = spec["template"]

    if template.count(CLAIM_SLOT) != 1:
        issues.append(f"template must contain exactly one {CLAIM_SLOT} placeholder")
    if template != template.strip():
        issues.append("template must not contain leading or trailing whitespace")
    if "\n" in template or "\r" in template:
        issues.append("template must remain single-line")

    slot_index = template.find(CLAIM_SLOT)
    if slot_index != -1:
        after_slot = template[slot_index + len(CLAIM_SLOT):]
        if after_slot.startswith((".", "?", "!", ",", ";", ":")):
            issues.append("template must not hard-code punctuation immediately after {claim}")

    rendered = build_fever_prompt('Example claim ending in "quotes."', spec["template_id"])
    if rendered != rendered.strip():
        issues.append("rendered prompt must not contain leading or trailing whitespace")
    if rendered.endswith((".", "?", "!", ",", ";", ":")):
        issues.append("rendered prompt must end with an answer slot, not punctuation")
    if "  " in rendered:
        issues.append("rendered prompt must not contain double spaces")

    return issues


def validate_fever_prompt_specs():
    validation_errors = {}
    for spec in FEVER_PROMPT_SPECS:
        issues = lint_fever_prompt_spec(spec)
        if issues:
            validation_errors[spec["template_id"]] = issues
    return validation_errors


def _normalize_category_filter(categories):
    if categories is None:
        return None
    if isinstance(categories, str):
        categories = [categories]
    categories = tuple(categories)
    unknown = sorted(set(categories) - set(_CATEGORY_SPECS))
    if unknown:
        available = ", ".join(_CATEGORY_SPECS)
        raise KeyError(f"Unknown FEVER prompt categories {unknown}. Available categories: {available}")
    return categories


def list_fever_prompt_categories():
    return [
        {"category_id": category_id, "category_label": category_label}
        for category_id, category_label in _CATEGORY_SPECS.items()
    ]


def iter_fever_prompt_specs(include_original=True, categories=None):
    category_filter = _normalize_category_filter(categories)
    for spec in FEVER_PROMPT_SPECS:
        if not include_original and spec["is_original"]:
            continue
        if category_filter is not None and spec["category_id"] not in category_filter:
            continue
        yield dict(spec)


def list_fever_prompt_template_ids(include_original=True, categories=None):
    return [spec["template_id"] for spec in iter_fever_prompt_specs(include_original=include_original, categories=categories)]


def get_fever_prompt_spec(template_id):
    if template_id not in FEVER_PROMPT_SPECS_BY_ID:
        available = ", ".join(list_fever_prompt_template_ids())
        raise KeyError(f"Unknown FEVER prompt template '{template_id}'. Available templates: {available}")
    return dict(FEVER_PROMPT_SPECS_BY_ID[template_id])


def build_fever_prompt(claim, template_id="original"):
    spec = get_fever_prompt_spec(template_id)
    normalized_claim = _normalize_claim(claim)
    return spec["template"].format(claim=normalized_claim)

# tests/test_untrusted_content.py
#
# The boundary between admin/customer-authored free text and an agent's
# context. Applied once, at the data contract (models/schemas.py), not at
# every call site.

from untrusted_content import mark_untrusted


def test_none_stays_none():
    assert mark_untrusted(None) is None


def test_plain_text_is_wrapped():
    assert mark_untrusted("Comfortable running shoes.") == (
        "<untrusted-user-content>Comfortable running shoes.</untrusted-user-content>"
    )


def test_control_characters_are_stripped():
    # \x07 is BEL, a C0 control character with no place in product text.
    assert mark_untrusted("Great\x07shoes") == (
        "<untrusted-user-content>Greatshoes</untrusted-user-content>"
    )


def test_bidi_override_characters_are_stripped():
    # U+202E (RIGHT-TO-LEFT OVERRIDE) is a documented technique for
    # visually disguising injected text -- it can make "reversed" text
    # read forwards on screen while parsing differently to a model.
    poisoned = "safe\u202edesc"
    result = mark_untrusted(poisoned)
    assert "\u202e" not in result
    assert result == "<untrusted-user-content>safedesc</untrusted-user-content>"


def test_tabs_and_newlines_survive():
    # Only control characters with no legitimate use in prose are
    # stripped -- ordinary formatting must not be mangled.
    assert mark_untrusted("Line one\nLine two\tend") == (
        "<untrusted-user-content>Line one\nLine two\tend</untrusted-user-content>"
    )


def test_long_text_is_truncated_at_the_cap():
    long_text = "a" * 5000
    result = mark_untrusted(long_text)
    # 4000 a's, then the truncation marker, then the closing tag.
    assert result == (
        "<untrusted-user-content>" + "a" * 4000 + "...[truncated]</untrusted-user-content>"
    )


def test_text_under_the_cap_is_not_truncated():
    text = "a" * 3999
    result = mark_untrusted(text)
    assert "...[truncated]" not in result
    assert result == f"<untrusted-user-content>{text}</untrusted-user-content>"

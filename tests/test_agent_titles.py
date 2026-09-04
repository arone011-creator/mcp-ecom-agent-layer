# tests/test_agent_titles.py
#
# A chat name is model-written text derived from an exchange that may
# have carried untrusted product copy, rendered into a narrow list. What
# it may CONTAIN is a smaller question than what it should say, and it is
# the one worth testing exhaustively.

from agent.titles import TITLE_LIMIT, clean_title


def test_a_plain_title_survives():
    assert clean_title("Recent order history") == "Recent order history"


def test_surrounding_quotes_are_removed():
    # Models like to answer a naming question in quotes.
    assert clean_title('"Recent order history"') == "Recent order history"
    assert clean_title("'Recent order history'") == "Recent order history"


def test_a_trailing_full_stop_is_removed():
    # A title is a label, not a sentence.
    assert clean_title("Recent order history.") == "Recent order history"


def test_newlines_become_spaces():
    # The list renders one line. A newline would silently lose the rest.
    assert clean_title("Recent\norder\nhistory") == "Recent order history"


def test_runs_of_whitespace_collapse():
    assert clean_title("Recent    order   history") == "Recent order history"


def test_a_url_is_stripped_out():
    # THE MUST NOT. The exchange this was derived from may have contained
    # untrusted product copy. A link in a chat name is never something the
    # customer asked for, and a name is rendered somewhere a message is
    # not.
    assert "http" not in clean_title("Deal at https://evil.example.com now")


def test_stripping_a_url_keeps_the_rest_of_the_title():
    # Stripped, not rejected: one bad link must not cost the name.
    assert clean_title("Lamp deal https://evil.example.com") == "Lamp deal"


def test_a_bare_www_link_is_stripped_too():
    assert "evil" not in clean_title("Lamp deal www.evil.example.com")


def test_a_long_title_is_cut_to_the_limit():
    assert len(clean_title("x" * 200)) <= TITLE_LIMIT


def test_a_title_that_is_only_a_url_is_refused():
    # Nothing left after stripping is not a name.
    assert clean_title("https://evil.example.com") is None


def test_an_empty_answer_is_refused():
    assert clean_title("") is None
    assert clean_title("   ") is None
    assert clean_title(None) is None


def test_a_non_string_is_refused():
    assert clean_title(["Recent order history"]) is None


def test_control_characters_are_removed():
    # A name goes into a list, a browser tab title and a log line.
    assert clean_title("Recent\x01\torder history") == "Recent order history"

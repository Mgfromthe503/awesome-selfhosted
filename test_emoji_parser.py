from emoji_parser import EMOJI_TRANSLATOR, EmojiParser


def test_parse_known_emoji():
    parser = EmojiParser(EMOJI_TRANSLATOR)
    details = parser.parse_emoji("🧄")
    assert details.meaning == "Root Chakra"


def test_parse_unknown_emoji():
    parser = EmojiParser(EMOJI_TRANSLATOR)
    details = parser.parse_emoji("❓")
    assert details.meaning == "Unknown Emoji"


def test_analyze_unknown_emoji_message():
    parser = EmojiParser(EMOJI_TRANSLATOR)
    assert parser.analyze_energy("❓") == "Cannot analyze unknown emoji."

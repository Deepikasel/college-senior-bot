from src.bot import SeniorBot

def test_basic():
    bot = SeniorBot.build("data/college_qa.jsonl", "models_test")
    bot.spice = 1
    out, dbg = bot.reply("How do I impress my professor?")
    assert isinstance(out, str) and len(out) > 0
    assert "Submit assignments on time" in out or "Tip:" in out

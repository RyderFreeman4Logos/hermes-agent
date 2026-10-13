"""Displayed reply-end line: measured time and exact last-response cache counts."""

from agent.reply_end_line import reply_end_line


def test_exact_hit_over_prompt_total_does_not_readd_cached_input():
    # prompt_total already contains cache_read; rate uses that same denominator.
    assert reply_end_line(elapsed_s=12.34, cache_read=800, prompt_total=1000) == (
        "12.3s · cache 800/1000 (80.0%)"
    )


def test_zero_hits_still_show_the_real_denominator():
    assert reply_end_line(elapsed_s=1, cache_read=0, prompt_total=50) == (
        "1.0s · cache 0/50 (0.0%)"
    )


def test_unknown_usage_omits_the_rate_instead_of_inventing_zero():
    assert reply_end_line(elapsed_s=2.0, cache_read=None, prompt_total=None) == "2.0s"
    assert reply_end_line(elapsed_s=2.0, cache_read=0, prompt_total=0) == "2.0s"
    assert reply_end_line(elapsed_s=None, cache_read=None, prompt_total=100) == ""

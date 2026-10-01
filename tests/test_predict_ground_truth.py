"""Ground truth is valid only for an unchanged historical input."""

from pages_src.predict import _matches_loaded_tweet


def test_tweet_ground_truth_is_invalidated_by_edit():
    loaded = {
        "tweet_text": "Wildfire reported near the highway",
        "tweet_keyword": "wildfire",
        "tweet_location": "California",
    }

    assert _matches_loaded_tweet(loaded.copy(), loaded)
    assert not _matches_loaded_tweet({**loaded, "tweet_text": "Just a movie"}, loaded)
    assert not _matches_loaded_tweet(loaded, {})

from scripts.d4_security import secret_error


def test_secret_error_redacts_before_truncating():
    message = "prefix " + "x" * 500 + " hf_secret_value"
    result = secret_error(RuntimeError(message))
    assert len(result) <= 160
    assert "hf_secret_value" not in result and "x" * 10 not in result

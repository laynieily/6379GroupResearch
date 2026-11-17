import pytest
import torch
import torch.nn.functional as F

from softmax_temp.evaluate import compute_metrics, ece
from softmax_temp.temp_softmax import TemperatureSoftmax, scaled_lr


def test_t1_matches_plain_softmax():
    logits = torch.randn(8, 10)
    assert torch.allclose(TemperatureSoftmax(1.0)(logits), F.softmax(logits, -1))


def test_temperature_sharpens_and_flattens():
    logits = torch.randn(4, 10)
    sharp = TemperatureSoftmax(0.1)(logits).max(-1).values
    flat = TemperatureSoftmax(10.0)(logits).max(-1).values
    assert (sharp > flat).all()


def test_argmax_invariant_to_temperature():
    logits = torch.randn(16, 10)
    for t in (0.1, 0.5, 1.0, 4.0):
        assert torch.equal(TemperatureSoftmax(t)(logits).argmax(-1), logits.argmax(-1))


def test_gradient_wrt_logits_scales_as_inverse_temperature():
    # d CE(logits/T)/d logits = (softmax(logits/T) - y) / T
    logits = torch.randn(1, 5, requires_grad=True)
    y = torch.tensor([2])
    t = 4.0
    TemperatureSoftmax(t).loss(logits, y).backward()
    expected = (F.softmax(logits.detach() / t, -1) - F.one_hot(y, 5)) / t
    assert torch.allclose(logits.grad, expected, atol=1e-6)


def test_invalid_temperature_rejected():
    with pytest.raises(ValueError):
        TemperatureSoftmax(0.0)


def test_scaled_lr_modes():
    assert scaled_lr(0.1, 4.0, "fixed") == 0.1
    assert scaled_lr(0.1, 4.0, "compensated", "sgd") == pytest.approx(0.4)
    assert scaled_lr(0.1, 4.0, "compensated", "adamw") == 0.1  # adaptive: no compensation
    with pytest.raises(ValueError):
        scaled_lr(0.1, 1.0, "tuned")


def test_ece_perfectly_calibrated_and_overconfident():
    probs = torch.tensor([[1.0, 0.0]] * 10)
    assert ece(probs, torch.zeros(10, dtype=torch.long)) == pytest.approx(0.0)  # always right
    assert ece(probs, torch.ones(10, dtype=torch.long)) == pytest.approx(1.0)   # always wrong, conf 1


def test_metrics_accuracy_temperature_invariant_but_nll_not():
    logits = torch.randn(64, 10)
    y = torch.randint(0, 10, (64,))
    a, b = compute_metrics(logits, y, 1.0), compute_metrics(logits, y, 4.0)
    assert a["acc"] == b["acc"]
    assert a["nll_T1"] == pytest.approx(b["nll_T1"])      # T1 metrics ignore train T
    assert a["nll_trainT"] != pytest.approx(b["nll_trainT"])

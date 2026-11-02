"""CPU regression checks for errors swallowed by BitBLAS's Python builder."""

import contextlib
import logging
import sys
import types

import pytest
import torch

from openpi.models_pytorch import bitblas_w8a8_layers as layers


@pytest.mark.parametrize("level", [logging.DEBUG, logging.WARNING])
def test_caught_bitblas_exception_keeps_full_traceback(caplog, level):
    logger = logging.getLogger("bitblas.ops.operator")
    original = (logger.level, logger.propagate, tuple(logger.handlers))
    complete_error = "compiler failure: " + "detail-" * 100 + "END_OF_ORIGINAL_ERROR"
    caplog.set_level(logging.DEBUG)

    with layers._capture_bitblas_errors("expert.layer.0", enabled=True) as capture:  # noqa: SLF001
        try:
            raise RuntimeError(complete_error)
        except RuntimeError:
            # Match upstream: debug/warning logs truncate/stringify the exception,
            # omit exc_info, then allow Matmul construction to continue.
            logger.log(level, "builder failed: %s", complete_error[:60])

        assert capture.error_count == 1
        assert complete_error in capture.first_error

    assert complete_error in caplog.text
    assert "Traceback (most recent call last)" in caplog.text
    assert "expert.layer.0" in caplog.text
    assert any(record.exc_info and record.levelno >= logging.ERROR for record in caplog.records)
    assert (logger.level, logger.propagate, tuple(logger.handlers)) == original


def test_capture_keeps_first_error_and_restores_logger_on_failure(caplog):
    logger = logging.getLogger("bitblas.ops.operator")
    initial_level = logger.level
    logger.setLevel(logging.ERROR)
    original = (logger.level, logger.propagate, tuple(logger.handlers))
    caplog.set_level(logging.DEBUG)

    def fail_after_logging():
        with layers._capture_bitblas_errors("expert.layer.1", enabled=True) as capture:  # noqa: SLF001
            for message in ("first schedule error", "later build error"):
                try:
                    raise RuntimeError(message)
                except RuntimeError:
                    logger.debug("upstream error")
            assert capture.error_count == 2
            assert "first schedule error" in capture.first_error
            assert "later build error" not in capture.first_error
            raise ValueError("outer failure")

    try:
        with pytest.raises(ValueError, match="outer failure"):
            fail_after_logging()
        assert (logger.level, logger.propagate, tuple(logger.handlers)) == original
    finally:
        logger.setLevel(initial_level)


def test_disabled_capture_does_not_change_bitblas_logging(caplog):
    logger = logging.getLogger("bitblas.ops.operator")
    original = (logger.level, logger.propagate, tuple(logger.handlers))
    with layers._capture_bitblas_errors("expert.layer.2", enabled=False):  # noqa: SLF001
        assert (logger.level, logger.propagate, tuple(logger.handlers)) == original
        try:
            raise RuntimeError("unobserved original exception")
        except RuntimeError:
            logger.debug("builder failed")
    assert "unobserved original exception" not in caplog.text
    assert (logger.level, logger.propagate, tuple(logger.handlers)) == original


def test_capture_preserves_warning_without_an_active_exception(caplog):
    logger = logging.getLogger("bitblas.ops.operator")
    caplog.set_level(logging.DEBUG)
    with layers._capture_bitblas_errors("expert.layer.3", enabled=True) as capture:  # noqa: SLF001
        logger.warning("no schedule found")
        assert capture.error_count == 0
        assert capture.first_error is None
    assert "no schedule found" in caplog.text


def test_malformed_upstream_log_does_not_mask_the_original_error(caplog):
    logger = logging.getLogger("bitblas.base.tuner")
    caplog.set_level(logging.DEBUG)
    with layers._capture_bitblas_errors("expert.layer.4", enabled=True) as capture:  # noqa: SLF001
        try:
            raise RuntimeError("original tuning failure")
        except RuntimeError as error:
            logger.debug("tuning failed", error)  # noqa: PLE1205 -- Reproduce BitBLAS's malformed log.
        assert capture.first_error == "RuntimeError: original tuning failure"
    assert "original tuning failure" in caplog.text
    assert "Traceback (most recent call last)" in caplog.text


class _PretendCudaTensor(torch.Tensor):
    """CPU storage whose device marker reaches initialization's runtime guard."""

    @property
    def device(self):
        return torch.device("cuda:0")


@pytest.mark.parametrize(("runtime_ready", "torch_ready"), [(False, False), (True, False), (False, True)])
def test_incomplete_runtime_stops_before_packing_without_changing_checkpoint(
    monkeypatch, caplog, runtime_ready, torch_ready
):
    layer = layers.BitBLASW8A8Linear(torch.nn.Linear(4, 3), "expert.layer.5")
    layer.qweight = torch.arange(12, dtype=torch.int8).reshape(3, 4).as_subclass(_PretendCudaTensor)
    layer.weight_scale.fill_(0.125)
    layer.bias.fill_(0.25)
    before = {name: value.clone() for name, value in layer.state_dict().items()}
    observed = {}

    class IncompleteMatmul:
        def __init__(self, config, **kwargs):
            observed.update(config=vars(config), **kwargs)
            self.rt_mod = object() if runtime_ready else None
            if torch_ready:
                self.torch_func = lambda: None
            try:
                raise RuntimeError("first scheduling failure with complete compiler detail")
            except RuntimeError:
                logging.getLogger("bitblas.ops.operator").debug("truncated compiler error")

        def transform_weight(self, _weight):
            pytest.fail("packing must not run after a swallowed build failure")

    bitblas = types.ModuleType("bitblas")
    ops = types.ModuleType("bitblas.ops")
    ops.Matmul = IncompleteMatmul
    ops.MatmulConfig = lambda **kwargs: types.SimpleNamespace(**kwargs)
    utils = types.ModuleType("bitblas.utils")
    utils.auto_detect_nvidia_target = lambda: "cuda -arch=sm_86"
    for name, module in (("bitblas", bitblas), ("bitblas.ops", ops), ("bitblas.utils", utils)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: contextlib.nullcontext())
    monkeypatch.setattr(layers, "_log_bitblas_environment", lambda *_args: None)
    monkeypatch.setenv("OPENPI_BITBLAS_DIAGNOSTICS", "1")
    caplog.set_level(logging.INFO)

    with pytest.raises(RuntimeError, match="stage=runtime_validation") as caught:
        layer.initialize_kernel(opt_m=(1, 16), enable_tuning=False)

    assert "first scheduling failure with complete compiler detail" in str(caught.value)
    assert "Traceback (most recent call last)" in caplog.text
    assert layer.kernel_stage == "runtime_validation"
    assert "incomplete TIR runtime" in layer.kernel_error
    assert not layer.kernel_verified
    assert layer.bitblas_matmul is None
    assert layer._packed_weight is None  # noqa: SLF001
    assert observed["backend"] == "tir"
    assert observed["enable_tuning"] is False
    assert observed["config"] == {
        "M": [1, 16],
        "N": 3,
        "K": 4,
        "A_dtype": "int8",
        "W_dtype": "int8",
        "out_dtype": "int32",
        "accum_dtype": "int32",
        "layout": "nt",
        "with_scaling": False,
        "with_bias": False,
    }
    assert before.keys() == layer.state_dict().keys()
    for name, value in layer.state_dict().items():
        assert torch.equal(value, before[name])

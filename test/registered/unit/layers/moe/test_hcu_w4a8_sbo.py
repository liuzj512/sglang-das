"""Unit tests for the HCU HIPC W4A8 SBO completion-signal contract."""

import ast
from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _load_w4a8_calls():
    # Execute the production functions without importing GPU-only MoE dependencies.
    source = Path(__file__).resolve().parents[5] / (
        "python/sglang/srt/layers/moe/ep_moe/layer.py"
    )
    tree = ast.parse(source.read_text())
    names = {"run_hcu_w4a8_hipc", "m_grouped_w4a8_gemm_nt_masked_hipc_wrapper"}
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    module = ModuleType("hcu_w4a8_sbo_test_target")
    module.import_module = import_module
    code = ast.Module(body=[ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    ), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), str(source), "exec"), module.__dict__)
    return module


hcu_w4a8_sbo = _load_w4a8_calls()


class TestHcuW4A8Sbo(CustomTestCase):
    def test_no_overlap_returns_output_and_disables_signal(self):
        out = object()
        gemm = Mock(return_value=(64, 24))
        with patch.object(hcu_w4a8_sbo, "import_module", return_value=SimpleNamespace(
            m_grouped_w4a8_gemm_nt_masked_hipc=gemm
        )):
            result = hcu_w4a8_sbo.run_hcu_w4a8_hipc((), (), out, object(), 12)
        self.assertIs(result, out)
        self.assertFalse(gemm.call_args.kwargs["enable_overlap"])
        self.assertIsNone(gemm.call_args.kwargs["signal"])

    def test_normal_wrapper_returns_tensor_instead_of_metadata(self):
        out = object()
        gemm = Mock(return_value=(64, 24))
        with patch.object(hcu_w4a8_sbo, "m_grouped_w4a8_gemm_nt_masked_hipc", gemm,
                          create=True):
            result = hcu_w4a8_sbo.m_grouped_w4a8_gemm_nt_masked_hipc_wrapper(
                object(), object(), object(), object(), out, object(), 12
            )
        self.assertIs(result, out)

    def test_hyv4_auxiliary_stream_is_created_for_rocm_or_sbo(self):
        root = Path(__file__).resolve().parents[5] / "python/sglang/srt/models"
        for name in ("hunyuan_v4.py", "hunyuan_v4_nextn.py"):
            tree = ast.parse((root / name).read_text())
            assigns = [node for node in ast.walk(tree)
                       if isinstance(node, ast.Assign) and any(
                           isinstance(t, ast.Attribute) and t.attr == "alt_stream"
                           and isinstance(t.value, ast.Name) and t.value.id == "self"
                           for t in node.targets)]
            self.assertEqual(len(assigns), 1)
            expression = compile(ast.Expression(assigns[0].value), name, "eval")
            for rocm, sbo in ((False, False), (True, False), (False, True)):
                with self.subTest(model=name, rocm=rocm, sbo=sbo):
                    stream = object()
                    create = Mock(return_value=stream)
                    namespace = {
                        "torch": SimpleNamespace(cuda=SimpleNamespace(Stream=create)),
                        "_is_cuda": False,
                        "envs": SimpleNamespace(
                            SGLANG_NPU_USE_MULTI_STREAM=SimpleNamespace(get=lambda: False),
                            SGLANG_ROCM_USE_MULTI_STREAM=SimpleNamespace(get=lambda: rocm),
                        ),
                        "is_sbo_enabled": lambda: sbo,
                    }
                    self.assertIs(eval(expression, namespace), stream if rocm or sbo else None)

    def test_overlap_records_event_and_metadata(self):
        out, signal = object(), object()
        calls = []
        gemm = Mock(return_value=(64, 24))
        module = SimpleNamespace(m_grouped_w4a8_gemm_nt_masked_hipc=gemm)
        args = SimpleNamespace(
            signal=signal,
            start_event=SimpleNamespace(record=lambda: calls.append("event")),
        )
        metadata = {}
        with patch.object(hcu_w4a8_sbo, "import_module", return_value=module):
            result = hcu_w4a8_sbo.run_hcu_w4a8_hipc(
                (), (), out, object(), 12, overlap_args=args, meta_overlap_args=metadata
            )
        self.assertIs(result, out)
        self.assertEqual(calls, ["event"])
        self.assertEqual(metadata, {"block_m": 64, "threshold": 24})
        self.assertTrue(gemm.call_args.kwargs["enable_overlap"])
        self.assertIs(gemm.call_args.kwargs["signal"], signal)

    def test_invalid_metadata_fails_closed(self):
        module = SimpleNamespace(
            m_grouped_w4a8_gemm_nt_masked_hipc=Mock(return_value=(64, 0))
        )
        args = SimpleNamespace(signal=object(), start_event=SimpleNamespace(record=lambda: None))
        with patch.object(hcu_w4a8_sbo, "import_module", return_value=module):
            with self.assertRaisesRegex(ValueError, "Invalid W4A8 HIPC completion metadata"):
                hcu_w4a8_sbo.run_hcu_w4a8_hipc(
                    (), (), object(), object(), 12, overlap_args=args, meta_overlap_args={}
                )

    def test_overlap_requires_shared_metadata(self):
        module = SimpleNamespace(
            m_grouped_w4a8_gemm_nt_masked_hipc=Mock(return_value=(64, 24))
        )
        with patch.object(hcu_w4a8_sbo, "import_module", return_value=module):
            with self.assertRaisesRegex(ValueError, "shared DeepEP metadata"):
                hcu_w4a8_sbo.run_hcu_w4a8_hipc(
                    (), (), object(), object(), 12, overlap_args=Mock()
                )


if __name__ == "__main__":
    import unittest

    unittest.main()

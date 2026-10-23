"""Check pair32 capability selection and the GEMM/DeepEP signal contract."""

import ast
import unittest
from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _load_fp8_calls():
    # Isolate the real signal/packing logic from GPU-only module imports.
    root = Path(__file__).resolve().parents[5] / "python/sglang/srt"
    targets = {
        "layers/moe/ep_moe/layer.py": {"run_hcu_fp8_pair32"},
        "layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8_moe.py": {
            "get_hcu_fp8_pair32_packer", "_prepare_dsv4_channel_fp8_deepgemm_weights"
        },
    }
    module = ModuleType("hcu_fp8_sbo_test_target")
    module.torch = torch
    module.import_module = import_module
    module.get_global_server_args = Mock(return_value=SimpleNamespace(deepep_mode="low_latency"))
    for path, names in targets.items():
        tree = ast.parse((root / path).read_text())
        nodes = [node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name in names]
        assert len(nodes) == len(names)
        code = ast.Module(body=[ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ), *nodes], type_ignores=[])
        exec(compile(ast.fix_missing_locations(code), path, "exec"), module.__dict__)
    return module


hcu_fp8_sbo = _load_fp8_calls()


class TestHcuFp8Sbo(CustomTestCase):
    def test_pair32_is_not_selected_for_normal_or_auto_mode(self):
        for mode in ("normal", "auto"):
            with self.subTest(mode=mode), patch.object(
                hcu_fp8_sbo, "get_global_server_args",
                return_value=SimpleNamespace(deepep_mode=mode)
            ), patch.object(torch.cuda, "get_device_properties") as properties:
                self.assertIsNone(hcu_fp8_sbo.get_hcu_fp8_pair32_packer(torch.device("cuda:0")))
                properties.assert_not_called()

    def test_packing_matches_the_selected_normal_or_masked_kernel(self):
        for mode, pair32 in (("normal", False), ("auto", False),
                             ("low_latency", False), ("low_latency", True)):
            with self.subTest(mode=mode, pair32=pair32):
                normal_pack = Mock(side_effect=lambda w: w.clone())
                legacy_pack = Mock(side_effect=lambda w: w.clone())
                pair32_pack = Mock(side_effect=lambda w: w.clone())
                weights = SimpleNamespace(w13_weight=torch.zeros(2, 4, 8),
                                          w2_weight=torch.zeros(2, 8, 4))
                owner = SimpleNamespace(_register_runtime_buffer=lambda layer, name, value:
                                        setattr(layer, name, value))
                modules = {
                    "deepgemm": ModuleType("deepgemm"),
                    "deepgemm.group_gemm_weight_pack": SimpleNamespace(
                        marlin_fp8_contiguous_weight=normal_pack),
                    "deepgemm.m_group_gemm": SimpleNamespace(
                        pack_int8_weight_enk_to_w6_low_latency_inplace=legacy_pack),
                }
                with patch.dict("sys.modules", modules), patch.object(
                    hcu_fp8_sbo, "get_global_server_args",
                    return_value=SimpleNamespace(deepep_mode=mode)
                ), patch.object(hcu_fp8_sbo, "get_hcu_fp8_pair32_packer",
                                return_value=pair32_pack if pair32 else None
                ), patch.object(torch.cuda, "empty_cache"):
                    hcu_fp8_sbo._prepare_dsv4_channel_fp8_deepgemm_weights(owner, weights)
                expected = pair32_pack if pair32 else legacy_pack if mode == "low_latency" else normal_pack
                self.assertEqual(expected.call_count, 2)
                self.assertEqual(sum(p.call_count for p in (normal_pack, legacy_pack, pair32_pack)), 2)
                self.assertEqual(weights._hcu_fp8_pair32, pair32)
                self.assertEqual(weights._dsv4_w13_weight_shape, (2, 4, 8))
                self.assertEqual(weights._dsv4_w2_weight_shape, (2, 8, 4))
                self.assertTrue(weights._dsv4_channel_fp8_deepgemm_repacked)
                self.assertFalse(hasattr(weights, "w13_weight"))
                self.assertFalse(hasattr(weights, "w2_weight"))

    def test_pair32_packing_is_specific_to_gfx938(self):
        pack = Mock()
        with (
            patch.object(torch.cuda, "get_device_properties") as properties,
            patch.object(hcu_fp8_sbo, "import_module", return_value=SimpleNamespace(
                pack_fp8_weight_pair32=pack
            )) as imports,
        ):
            self.assertIsNone(hcu_fp8_sbo.get_hcu_fp8_pair32_packer(torch.device("cpu")))
            properties.return_value.gcnArchName = "gfx936"
            self.assertIsNone(hcu_fp8_sbo.get_hcu_fp8_pair32_packer(torch.device("cuda:0")))
            imports.assert_not_called()
            properties.return_value.gcnArchName = "gfx938:sramecc+"
            self.assertIs(hcu_fp8_sbo.get_hcu_fp8_pair32_packer(torch.device("cuda:0")), pack)

    def test_completion_metadata_comes_from_selected_kernel(self):
        out, signal, counts = object(), object(), object()
        for block_m in (64, 128):
            with self.subTest(block_m=block_m):
                calls = []
                gemm = Mock(side_effect=lambda *a, **kw: (calls.append("gemm") or (block_m, 24)))
                args = SimpleNamespace(
                    signal=signal,
                    start_event=SimpleNamespace(record=lambda: calls.append("event")),
                )
                meta = {}
                with patch.object(hcu_fp8_sbo, "import_module", return_value=SimpleNamespace(
                    m_grouped_fp8_gemm_nt_masked_pair32=gemm
                )):
                    result = hcu_fp8_sbo.run_hcu_fp8_pair32(
                        (), (), out, counts, 7, overlap_args=args, meta_overlap_args=meta
                    )
                self.assertIs(result, out)
                self.assertEqual(calls, ["event", "gemm"])
                self.assertEqual(meta, {"block_m": block_m, "threshold": 24})
                self.assertIs(gemm.call_args.kwargs["signal"], signal)
                self.assertTrue(gemm.call_args.kwargs["enable_overlap"])

    def test_no_overlap_passes_valid_tensor_placeholder(self):
        counts, out = object(), object()
        gemm = Mock(return_value=(64, 3))
        with patch.object(hcu_fp8_sbo, "import_module", return_value=SimpleNamespace(
            m_grouped_fp8_gemm_nt_masked_pair32=gemm
        )):
            hcu_fp8_sbo.run_hcu_fp8_pair32((), (), out, counts, 7)
            with self.assertRaisesRegex(ValueError, "shared DeepEP metadata"):
                hcu_fp8_sbo.run_hcu_fp8_pair32((), (), out, counts, 7, overlap_args=Mock())
        self.assertEqual(gemm.call_count, 1)
        self.assertIs(gemm.call_args.kwargs["signal"], counts)
        self.assertFalse(gemm.call_args.kwargs["enable_overlap"])


if __name__ == "__main__":
    unittest.main()

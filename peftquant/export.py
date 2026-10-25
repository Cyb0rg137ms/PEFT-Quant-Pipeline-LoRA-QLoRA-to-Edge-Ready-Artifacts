"""Quantise whole models and export/load verified edge artefacts.

An artefact is a directory with

* ``weights.pt``: packed integer codes, scales and zero-points for every quantised
  layer, plus the remaining float tensors (biases, norms, embeddings);
* ``manifest.json``: quantisation spec per layer, byte counts, SHA-256 of
  ``weights.pt``, and the verification results recorded at export time
  (output error on probe inputs against the float reference model).

:func:`load_artifact` checks the hash before rebuilding the model.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn

from peftquant.gptq import gptq_quantize, hessian
from peftquant.quant import QTensor, QuantLinear, QuantSpec, quantize

log = logging.getLogger(__name__)
Method = Literal["rtn", "gptq"]


def _linear_children(model: nn.Module) -> list[tuple[nn.Module, str, str, nn.Linear]]:
    out = []
    for name, module in model.named_modules():
        for cname, child in module.named_children():
            if type(child) is nn.Linear:
                out.append((module, cname, f"{name}.{cname}" if name else cname, child))
    return out


@torch.no_grad()
def quantize_model(
    model: nn.Module,
    spec: QuantSpec = QuantSpec(),
    method: Method = "rtn",
    calib: Tensor | None = None,
    skip: tuple[str, ...] = (),
) -> nn.Module:
    """Return a copy of ``model`` with every ``nn.Linear`` (except ``skip``) replaced by :class:`QuantLinear`.

    ``gptq`` needs ``calib`` inputs; layer inputs are captured with forward
    hooks on the *float* model in one pass (not the sequential variant).
    """
    q = copy.deepcopy(model).eval()
    targets = [t for t in _linear_children(q) if t[2] not in skip]
    inputs: dict[str, list[Tensor]] = {}
    if method == "gptq":
        if calib is None:
            raise ValueError("gptq needs calibration inputs")
        hooks = [
            lin.register_forward_hook(lambda _m, inp, _o, n=name: inputs.setdefault(n, []).append(inp[0].detach()))
            for _, _, name, lin in targets
        ]
        try:
            q(calib)
        finally:
            for h in hooks:
                h.remove()
    for parent, cname, name, lin in targets:
        if method == "gptq":
            H = hessian(torch.cat([x.reshape(-1, lin.in_features) for x in inputs[name]]))
            qt = gptq_quantize(lin.weight, H, spec)
        else:
            qt = quantize(lin.weight, spec)
        setattr(parent, cname, QuantLinear(qt, lin.bias))
        log.debug("quantised %s with %s", name, method)
    return q


def model_nbytes(model: nn.Module) -> int:
    """Stored bytes: QuantLinear payloads plus fp32 bytes of every other parameter/buffer."""
    total = 0
    quant_params: set[int] = set()
    for m in model.modules():
        if isinstance(m, QuantLinear):
            total += m.nbytes
            if m.bias is not None:
                quant_params.add(id(m.bias))
    for t in list(model.parameters()) + list(model.buffers()):
        if id(t) not in quant_params:
            total += t.numel() * 4
    return total


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def export_artifact(
    qmodel: nn.Module,
    path: str | Path,
    reference: nn.Module | None = None,
    probe: Tensor | None = None,
) -> dict:
    """Write ``weights.pt`` + ``manifest.json``; returns the manifest."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    layers: dict[str, dict] = {}
    tensors: dict[str, Tensor] = {}
    qnames = set()
    for name, m in qmodel.named_modules():
        if isinstance(m, QuantLinear):
            qt = m.qweight
            qnames.add(name)
            tensors[f"{name}.codes"] = qt.codes
            tensors[f"{name}.scale"] = qt.scale.half()
            tensors[f"{name}.zero"] = qt.zero.half()
            if m.bias is not None:
                tensors[f"{name}.bias"] = m.bias.detach()
            layers[name] = {
                "shape": list(qt.shape),
                "packed": qt.packed,
                "spec": asdict(qt.spec),
                "bytes": m.nbytes,
            }
    for k, v in qmodel.state_dict().items():
        if k.rsplit(".", 1)[0] not in qnames:
            tensors[k] = v
    wpath = out / "weights.pt"
    torch.save(tensors, wpath)
    manifest: dict = {
        "format": "peftquant-artifact/1",
        "layers": layers,
        "sha256": _sha256(wpath),
        "payload_bytes": model_nbytes(qmodel),
        "file_bytes": wpath.stat().st_size,
    }
    if reference is not None and probe is not None:
        ref, got = reference.eval()(probe), qmodel.eval()(probe)
        manifest["verification"] = {
            "probe_shape": list(probe.shape),
            "max_abs_err": float((ref - got).abs().max()),
            "rel_err": float((ref - got).norm() / ref.norm().clamp_min(1e-30)),
        }
        if ref.dim() == 2 and ref.shape[1] > 1:
            manifest["verification"]["top1_agreement"] = float((ref.argmax(1) == got.argmax(1)).float().mean())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_artifact(path: str | Path, skeleton: nn.Module) -> nn.Module:
    """Rebuild a quantised model from an artefact onto a float ``skeleton`` of the same architecture.

    Raises ``ValueError`` if the weights file does not match the manifest hash.
    """
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text())
    wpath = root / "weights.pt"
    if _sha256(wpath) != manifest["sha256"]:
        raise ValueError("weights.pt hash mismatch: artefact corrupted or tampered with")
    tensors = torch.load(wpath, map_location="cpu", weights_only=True)
    model = copy.deepcopy(skeleton).eval()
    for name, info in manifest["layers"].items():
        parent_name, _, cname = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        spec = QuantSpec(**info["spec"])
        qt = QTensor(
            tensors[f"{name}.codes"],
            tensors[f"{name}.scale"].float(),
            tensors[f"{name}.zero"].float(),
            tuple(info["shape"]),
            spec,
            info["packed"],
        )  # type: ignore[arg-type]
        setattr(parent, cname, QuantLinear(qt, tensors.get(f"{name}.bias")))
    rest = {k: v for k, v in tensors.items() if k.rsplit(".", 1)[0] not in manifest["layers"]}
    _missing, unexpected = model.load_state_dict(rest, strict=False)
    if unexpected:
        raise ValueError(f"unexpected tensors in artefact: {unexpected}")
    return model

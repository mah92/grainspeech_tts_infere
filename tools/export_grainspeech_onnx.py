"""Export the trained GrainSpeech checkpoint to ONNX for the CPU-only C++ inference program.

Model: GrainSpeech acoustic model (phones -> mel). Inputs:
    phoneme      int64   [1, N]
    phoneme_mask bool    [1, N]      (all False = nothing padded)
Output:
    mel          float32 [1, T, 80]  (the same space the Vocos expects — no renormalisation)

Also writes the symbol table (id = line number, python-style) so the C++ side can tokenise with
exactly the same ids as `text_to_sequence`.

Usage: python export_grainspeech_onnx.py --ckpt <ckpt> --config <preprocess.yaml> --out <onnx> \
        --symbols-out <txt> [--sr 16000]
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

FORK = Path(__file__).resolve().parent
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(FORK / "grainspeech"))
os.chdir(FORK)

from model_l1_ssim_gvar import GrainSpeech  # noqa: E402
from text.symbols import symbols  # noqa: E402


class MelOnly(torch.nn.Module):
    """Wraps the acoustic model so ONNX sees a single (ids, mask) -> mel function."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, phoneme, phoneme_mask):
        batch = {"phoneme": phoneme, "phoneme_mask": phoneme_mask}
        with torch.inference_mode():
            mel, mel_len, _dur = self.model.phoneme2mel(batch, train=False)
        return mel, mel_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--symbols-out", required=True)
    ap.add_argument("--opset", type=int, default=15)
    args = ap.parse_args()

    with open(args.config) as fh:
        pre = yaml.safe_load(fh)
    pre["path"]["preprocessed_path"] = str(FORK / pre["path"]["preprocessed_path"])

    model = GrainSpeech.load_from_checkpoint(
        args.ckpt, map_location="cpu", preprocess_config=pre,
        hifigan_checkpoint="none", infer_device="cpu").eval()
    wrapper = MelOnly(model).eval()

    # Export with a batch of 1: with B>1 the batch loop is unrolled and the graph only accepts
    # that exact batch. With B=1 phoneme_mask is unused (the model only applies it for batch>1),
    # so it gets pruned — fine for synthesis, which is one sequence at a time.
    n, b = 20, 1
    ids = torch.randint(1, len(symbols) - 1, (b, n), dtype=torch.long)
    mask = torch.zeros((b, n), dtype=torch.bool)

    torch.onnx.export(
        wrapper, (ids, mask), args.out,
        input_names=["phoneme", "phoneme_mask"],
        output_names=["mel", "mel_len"],
        dynamic_axes={"phoneme": {0: "B", 1: "N"}, "phoneme_mask": {0: "B", 1: "N"},
                      "mel": {0: "B", 1: "T"}, "mel_len": {0: "B"}},
        opset_version=args.opset, do_constant_folding=True,
    )
    print(f"exported {args.out} ({Path(args.out).stat().st_size/1e6:.2f} MB)")

    Path(args.symbols_out).write_text("\n".join(symbols) + "\n", encoding="utf-8")
    print(f"symbol table: {len(symbols)} ids -> {args.symbols_out} "
          f"(pad id {symbols.index('_pad_') if '_pad_' in symbols else 0})")

    # numeric check: ONNX vs torch on the same input
    import numpy as np
    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    print("onnx inputs:", [i.name for i in sess.get_inputs()],
          "| outputs:", [o.name for o in sess.get_outputs()])
    feed = {i.name: (ids.numpy() if i.name == "phoneme" else mask.numpy())
            for i in sess.get_inputs()}
    onx = sess.run(None, feed)
    # dynamic-length check: several sequence lengths must all work
    for n_test in (7, 33, 61):
        it = torch.randint(1, len(symbols) - 1, (1, n_test), dtype=torch.long)
        mt = torch.zeros((1, n_test), dtype=torch.bool)
        feed_t = {i.name: (it.numpy() if "phoneme" == i.name else mt.numpy())
                  for i in sess.get_inputs()}
        o = sess.run(None, feed_t)
        print(f"  dynamic N={n_test}: mel {o[0].shape}")
    with torch.inference_mode():
        ref_mel, ref_len, _ = model.phoneme2mel({"phoneme": ids, "phoneme_mask": mask}, train=False)
    a = np.asarray(onx[0]); b = ref_mel.numpy()
    t = min(a.shape[1], b.shape[1])
    print(f"check: onnx mel {a.shape} vs torch {b.shape} | max |diff| {np.abs(a[:, :t]-b[:, :t]).max():.2e} "
          f"| corr {np.corrcoef(a[:, :t].ravel(), b[:, :t].ravel())[0,1]:.6f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
OpenVINO conversion template (Lite skill).
Adapt to the specific model: try direct export first, fall back to ONNX.
"""
import openvino as ov

# ====================== Configure for the specific model ======================
OUTPUT_XML = "converted_model.xml"   # output path, .bin is written alongside it
INPUT_SHAPE = [1, 3, 224, 224]       # example_input shape for tracing
# =================================================================================


def convert_direct(model, example_input):
    """Path A: direct torch/TF module -> OpenVINO IR."""
    ov_model = ov.convert_model(model, example_input=example_input)
    ov.save_model(ov_model, OUTPUT_XML, compress_to_fp16=True)
    return ov_model


def convert_via_onnx(onnx_path):
    """Path B: ONNX intermediate -> OpenVINO IR (fallback if direct export fails)."""
    ov_model = ov.convert_model(onnx_path)
    ov.save_model(ov_model, OUTPUT_XML, compress_to_fp16=True)
    return ov_model


def main():
    import torch

    # Replace with the actual model class + weight loading for this repo.
    # model = MyModel()
    # model.load_state_dict(torch.load("checkpoint.pt", map_location="cpu"))
    # model.eval()
    # example_input = torch.randn(*INPUT_SHAPE)

    raise NotImplementedError(
        "Fill in model construction, weight loading, and example_input, "
        "then call convert_direct(model, example_input); if that raises, "
        "export ONNX with torch.onnx.export(...) and call convert_via_onnx(path)."
    )


if __name__ == "__main__":
    main()

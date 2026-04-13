"""
Test mosaicml-streaming + transformers compatibility on SGC.
"""

import sys


def check_versions():
    print(f"Python: {sys.version}\n")

    import torch
    print(f"torch:              {torch.__version__}")
    print(f"CUDA available:     {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device:        {torch.cuda.get_device_name(0)}")

    import transformers
    print(f"transformers:       {transformers.__version__}")

    import streaming
    print(f"mosaicml-streaming: {streaming.__version__}")


def test_streaming():
    """Basic streaming dataset import test."""
    from streaming import StreamingDataset
    print("\n[streaming] StreamingDataset import OK")


if __name__ == "__main__":
    print("=" * 50)
    print("Streaming + Transformers Compatibility Test")
    print("=" * 50 + "\n")

    check_versions()
    test_streaming()

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)

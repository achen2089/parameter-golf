#!/usr/bin/env python3
"""Build compressed train_gpt.py from train_gpt_human.py.

The submission scoring counts Path(__file__).read_text() as code_bytes. Raw
source is ~60KB; after lzma+base85 it's ~18KB, saving ~42KB toward the 16MB
artifact limit. The reference SOTA submissions do the same.

Usage:
    python3 build_train_gpt.py
"""
import base64
import lzma
import pathlib

SRC = pathlib.Path(__file__).parent / "train_gpt_human.py"
DST = pathlib.Path(__file__).parent / "train_gpt.py"

def main() -> None:
    source = SRC.read_bytes()
    compressed = lzma.compress(source, preset=9)
    b85 = base64.b85encode(compressed).decode("ascii")
    wrapper = (
        "import lzma as L,base64 as B\n"
        f'exec(L.decompress(B.b85decode("{b85}")))\n'
    )
    DST.write_text(wrapper)
    print(f"train_gpt_human.py: {len(source):>6,} bytes (source)")
    print(f"train_gpt.py:       {len(wrapper):>6,} bytes (compressed)")
    print(f"saved:              {len(source) - len(wrapper):>6,} bytes "
          f"({100*(1 - len(wrapper)/len(source)):.1f}% smaller)")

if __name__ == "__main__":
    main()

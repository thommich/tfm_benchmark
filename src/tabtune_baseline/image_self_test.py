"""Build-time package assertions; host driver visibility is checked by the worker."""

from __future__ import annotations

import importlib.metadata
import sys


def main() -> None:
    import numpy
    import pandas
    import sklearn
    import torch

    assert sys.version_info[:2] == (3, 11), sys.version
    assert importlib.metadata.version("tabtune") == "0.2.1"
    assert torch.__version__.startswith("2.7.1") and torch.version.cuda == "11.8"
    assert int(torch.backends.cudnn.version() or 0) // 1000 == 9
    assert pandas.__version__ == "2.3.3", pandas.__version__
    assert numpy.__version__ == "2.4.6", numpy.__version__
    assert sklearn.__version__ == "1.9.1", sklearn.__version__
    for name, module in (("pandas", pandas), ("numpy", numpy), ("scikit-learn", sklearn)):
        print(f"{name}={module.__version__}")
    print(f"python={sys.version}; torch={torch.__version__}; cuda={torch.version.cuda}; "
          f"cudnn={torch.backends.cudnn.version()}; tabtune=0.2.1")


if __name__ == "__main__":
    main()

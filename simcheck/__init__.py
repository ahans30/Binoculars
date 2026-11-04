import os
from pathlib import Path

# 索引には購読誌PDFの本文も入るため、OneDriveではなくプロジェクト直下のローカルに置く
DATA_DIR = Path(os.environ.get("SIMCHECK_DATA_DIR") or Path(__file__).resolve().parent.parent / "data" / "simcheck")

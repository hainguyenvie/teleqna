"""Gọi CLI của Inspect AI mà không cần venv: image cài thư viện bằng
`pip install --target /opt/eval/pylibs`, nên entry point script không tồn tại."""
from inspect_ai._cli.main import main

main()

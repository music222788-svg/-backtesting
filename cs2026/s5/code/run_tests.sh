#!/bin/bash
# S5 검증 일괄 실행: 기존 테스트는 원본을 건드리지 않도록 임시 사본에서 실행한다.
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
TMP="$(mktemp -d)"
cp -r "$ROOT/code" "$ROOT/data" "$ROOT/out" "$TMP/"
cp "$ROOT/s4/rebuilt/s1_macro.csv" "$ROOT/s4/rebuilt/s1_proxy_returns.csv" "$TMP/out/"
(cd "$TMP" && python3 code/test_s2_timing.py | grep -E "PASS$|FAIL" | tail -1)
(cd "$TMP" && python3 code/test_s3_fast.py | grep "불일치")
rm -rf "$TMP"
python3 "$HERE/code/test_s5_dca.py" | tail -1

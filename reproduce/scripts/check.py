import json, sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
EXPECTED = json.loads((Path(__file__).resolve().parents[1] / 'expected.json').read_text())


def printed(value):
    return float(Decimal(f'{value:.2f}').quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))


def check(section, got):
    ok = True
    print(f'\n{section}: MAPE % (reproduced -> two decimals -> one decimal, half-up, as printed in the paper)')
    for name, want in EXPECTED[section].items():
        n, mape = got[name]
        good = n == want['n'] and printed(mape) == want['mape']
        ok &= good
        print(f"  {name:18s} n={n:4d} (paper {want['n']:3d})  {mape:7.3f} -> {mape:6.2f} -> {printed(mape):5.1f}  paper {want['mape']:5.1f}  {'OK' if good else 'MISMATCH'}")
    return ok


if __name__ == '__main__':
    sys.exit(0)

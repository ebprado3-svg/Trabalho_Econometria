"""Confere a soma de famílias e as médias monetárias no PDF municipal original."""
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

def valor(s):
    return Decimal(s.replace(".", "").replace(",", "."))

padrao = re.compile(r"^\s*([A-Z]{2})\s+(.+?)\s+([\d.]+)\s+R\$\s+([\d.,]+)\s+R\$\s+([\d.,]+)")
municipios = familias = divergencias = 0
for linha in Path("folha.txt").read_text().splitlines():
    m = padrao.match(linha)
    if m:
        n, total, media = map(valor, m.group(3, 4, 5))
        municipios += 1
        familias += int(n)
        divergencias += (total/n).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) != media
assert (municipios, familias, divergencias) == (5570, 20653849, 2458)
print(f"{municipios} municípios, {familias} famílias e {divergencias} médias divergentes.")


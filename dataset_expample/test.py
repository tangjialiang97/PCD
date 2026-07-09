
import re
from pathlib import Path
import pandas as pd

def dec(n):
    return re.sub(r'#U([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1),16)), n)

p = Path('.')

p = Path('.')
for f in sorted(p.glob('*.xlsx')):
    print('='*70)
    print(dec(f.name), '|', f.name)
    xl = pd.ExcelFile(f)
    for s in xl.sheet_names:
        df = xl.parse(s)
        print(' sheet', s, df.shape)
        print(' cols', list(df.columns))
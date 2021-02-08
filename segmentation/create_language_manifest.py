""" A script to automatically generate MANIFEST.in files for different languages packages language
"""
import sys

MANIFEST_FILE = 'MANIFEST.in'
language_code = sys.argv[1]

with open(MANIFEST_FILE, 'w') as f:
    f.write('global-exclude *.ipynb\n')
    f.write('global-exclude *.pyc\n')
    f.write('global-exclude *.md\n')
    f.write(f'graft qsegmt/{language_code}\n')
    f.write(f'prune qsegmt/{language_code}/__pycache__\n')

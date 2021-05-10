import os

'''-----------------------------------------------------------------------------
DIRECTORIES AND FILES
-----------------------------------------------------------------------------'''
MODULE_DIR = os.path.dirname(os.path.realpath(__file__))
PY_PROJECT_DIR = os.path.join(MODULE_DIR, '..')
PROJECT_DIR = os.path.join(PY_PROJECT_DIR, '..')
DB_DIR = os.path.join(PROJECT_DIR, 'data')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')
EVAL_DIR = os.path.join(OUTPUT_DIR, 'evaluation')


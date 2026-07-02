"""
EaR³T — Installation Verifier
==============================
Run this right after completing the install steps to confirm every component
is working before the summer school examples.

Usage
-----
    cd $ERTDIR/er3t-edu/er3t/examples
    conda activate er3t
    python verify_install.py              # core checks
    python verify_install.py --worldview  # also test Worldview image download

Expected outcome
----------------
All checks should show  ✓  except abs_16g, which is marked  —  (not included
in the teaching data package and not needed for examples 01–05).

The Worldview test requires internet access and takes ~30 s; run it once to
confirm network connectivity before the summer school session.
"""

import os
import sys
import argparse
import datetime

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='EaR³T installation verifier')
parser.add_argument('--worldview', action='store_true',
                    help='Also test Worldview RGB image download (~30 s, needs internet)')
args = parser.parse_args()

# ── Terminal colours (gracefully degraded if terminal doesn't support ANSI) ───
_use_colour = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
def _c(code, text): return ('\033[%sm' % code + text + '\033[0m') if _use_colour else text
def _green(t):  return _c('92', t)
def _red(t):    return _c('91', t)
def _yellow(t): return _c('93', t)
def _bold(t):   return _c('1',  t)

# ── Result trackers ───────────────────────────────────────────────────────────
_results = []   # list of (label, status, note)  status in 'pass','fail','skip'

def _ok(label, note=''):
    _results.append((label, 'pass', note))
    print('  %s  %s%s' % (_green('✓'), label, ('  — ' + note) if note else ''))

def _fail(label, hint=''):
    _results.append((label, 'fail', hint))
    print('  %s  %s' % (_red('✗'), label))
    if hint:
        print('       %s' % _yellow('→ ' + hint))

def _skip(label, note=''):
    _results.append((label, 'skip', note))
    print('  %s  %s%s' % (_yellow('—'), label, ('  (' + note + ')') if note else ''))

def _section(title):
    print('\n' + _bold(title))


# ══════════════════════════════════════════════════════════════════════════════
print()
print(_bold('╭────────────────────────────────────────────────╮'))
print(_bold('       EaR³T  —  Installation Verifier            '))
print(_bold('╰────────────────────────────────────────────────╯'))


# ── 1. Python version ─────────────────────────────────────────────────────────
_section('1. Python environment')
vi = sys.version_info
if vi.major == 3 and vi.minor >= 9:
    _ok('Python %d.%d.%d' % (vi.major, vi.minor, vi.micro))
else:
    _fail('Python %d.%d.%d' % (vi.major, vi.minor, vi.micro),
          'Python 3.9+ recommended')


# ── 2. Core imports ───────────────────────────────────────────────────────────
_section('2. Package imports')

try:
    import numpy as np
    _ok('numpy %s' % np.__version__)
except ImportError as e:
    _fail('numpy', str(e))

try:
    import scipy
    _ok('scipy %s' % scipy.__version__)
except ImportError as e:
    _fail('scipy', str(e))

try:
    import matplotlib
    _ok('matplotlib %s' % matplotlib.__version__)
except ImportError as e:
    _fail('matplotlib', str(e))

try:
    import h5py
    _ok('h5py %s' % h5py.__version__)
except ImportError as e:
    _fail('h5py', str(e))

try:
    import netCDF4
    _ok('netCDF4 %s' % netCDF4.__version__)
except ImportError as e:
    _fail('netCDF4', str(e))

try:
    import er3t
    _ok('er3t  (installed at %s)' % os.path.dirname(er3t.__file__))
except ImportError as e:
    _fail('er3t', 'pip install -e . not run, or conda env not active')
    print('\nCannot continue without er3t — exiting.')
    sys.exit(1)


# ── 3. Atmosphere module ──────────────────────────────────────────────────────
_section('3. Atmosphere module  (er3t.pre.atm)')

try:
    import numpy as np
    from er3t.pre.atm import atm_atmmod
    levels = np.linspace(0.0, 20.0, 41)
    atm0   = atm_atmmod(levels=levels)
    _ok('atm_atmmod — US standard atmosphere, 41 levels')
except Exception as e:
    atm0 = None
    _fail('atm_atmmod', str(e))


# ── 4. Absorption: abs_16g ────────────────────────────────────────────────────
_section('4. Absorption: correlated-k 16g  (er3t.pre.abs.abs_16g)')

try:
    import er3t.common
    h5_path = '%s/abs_16g.h5' % er3t.common.fdir_data_abs
    if not os.path.exists(h5_path):
        _skip('abs_16g',
              'abs_16g.h5 not in data package — not needed for examples 01–05')
    else:
        from er3t.pre.abs import abs_16g
        abs0 = abs_16g(wavelength=500.0, atm_obj=atm0)
        _ok('abs_16g  @ 500 nm')
except Exception as e:
    _fail('abs_16g', str(e))


# ── 5. Absorption: REPTRAN ────────────────────────────────────────────────────
_section('5. Absorption: REPTRAN  (er3t.pre.abs.abs_rep)')

try:
    from er3t.pre.abs import abs_rep
    if atm0 is None:
        _skip('abs_rep', 'skipped because atm_atmmod failed')
    else:
        abs1 = abs_rep(wavelength=500.0, target='coarse', atm_obj=atm0)
        _ok('abs_rep  @ 500 nm, target=coarse')
except Exception as e:
    _fail('abs_rep', str(e))


# ── 6. Mie phase function ─────────────────────────────────────────────────────
_section('6. Mie phase function  (er3t.pre.pha.pha_mie_wc)')

try:
    from er3t.pre.pha import pha_mie_wc
    import numpy as np
    # Use a coarse angle grid to keep it fast
    angles = np.arange(0.0, 181.0, 1.0)
    pha0   = pha_mie_wc(wavelength=500.0, angles=angles)
    g      = pha0.data['asy']['data'].mean()
    _ok('pha_mie_wc  @ 500 nm  (mean asymmetry g = %.3f)' % g)
except Exception as e:
    _fail('pha_mie_wc', str(e))


# ── 7. MCARaTS executable ─────────────────────────────────────────────────────
_section('7. MCARaTS solver')

mcarats_exe = os.environ.get('MCARATS_V010_EXE', '')
if not mcarats_exe:
    _fail('MCARATS_V010_EXE',
          'environment variable not set — add to your shell rc file:\n'
          '       export MCARATS_V010_EXE="$ERTDIR/mcarats-0.10.3/src/mcarats"')
elif not os.path.isfile(mcarats_exe):
    _fail('MCARATS_V010_EXE points to missing file',
          'Path: %s\n       Check MCARaTS build (Step 3 of install guide).' % mcarats_exe)
elif not os.access(mcarats_exe, os.X_OK):
    _fail('MCARaTS binary not executable', mcarats_exe)
else:
    _ok('MCARaTS executable found: %s' % mcarats_exe)


# ── 8. Worldview download (optional) ─────────────────────────────────────────
if args.worldview:
    _section('8. Worldview RGB download  (optional, ~30 s)')
    try:
        from er3t.util.daac import download_worldview_image
        # Small region, single satellite/instrument — quick smoke test
        date   = datetime.datetime(2022, 5, 18)
        extent = [-92.0, -90.0, 33.0, 35.0]   # small ~200 km patch over US
        fname  = download_worldview_image(
            date, extent,
            fdir_out='tmp-data/verify',
            instrument='modis',
            satellite='aqua',
            fmt='png',
        )
        if fname and os.path.exists(fname):
            _ok('download_worldview_image — saved to %s' % fname)
        else:
            _fail('download_worldview_image', 'function returned but file not found')
    except Exception as e:
        _fail('download_worldview_image', str(e))
else:
    _section('8. Worldview RGB download  (optional)')
    _skip('skipped — re-run with  --worldview  to test network/Worldview access')


# ── Summary ───────────────────────────────────────────────────────────────────
print()
print(_bold('─' * 50))
n_pass = sum(1 for _, s, _ in _results if s == 'pass')
n_fail = sum(1 for _, s, _ in _results if s == 'fail')
n_skip = sum(1 for _, s, _ in _results if s == 'skip')
print(_bold('Summary: %s  %s  %s' % (
    _green('%d passed' % n_pass),
    _red('%d failed' % n_fail) if n_fail else '0 failed',
    '%d skipped' % n_skip,
)))

if n_fail == 0:
    print()
    print(_green(_bold('All required checks passed — your er3t installation is ready!')))
    print('You can now run the example scripts in this directory (01–05).')
else:
    print()
    print(_red(_bold('%d check(s) failed — see hints above before proceeding.' % n_fail)))
    print('Consult the install guide at https://konradsebastian.github.io/er3t-edu/install.html')
print()

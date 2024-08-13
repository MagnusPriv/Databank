from unittest import mock
import sys, os
from urllib.error import HTTPError
import pytest

import DatabankLib

@pytest.fixture(autouse=True, scope="session")
def header_session_scope():
    _rp = os.path.join(os.path.dirname(__file__), "Data", "Simulations.2")
    with mock.patch.object(DatabankLib, "NMLDB_SIMU_PATH", _rp):
        print("DBG: Mocking simulation path: ", DatabankLib.NMLDB_SIMU_PATH)
        yield
    print("DBG: Mocking completed")

@pytest.fixture(scope="module")
def systems():
    from DatabankLib.core import initialize_databank
    s = initialize_databank()
    print(f"Loaded: {len(s)} systems")
    yield s

def test_initialize_n(systems):
    assert len(systems) == 5

def test_print_README(systems, capsys):
    from DatabankLib.core import print_README
    sys0 = systems[0]
    print_README(sys0)
    output: str = capsys.readouterr().out.rstrip()
    fDOI = output.find('DOI:') != -1
    fTEMP = output.find('TEMPERATURE:') != -1
    assert fDOI and fTEMP

@pytest.mark.parametrize("systemid, result", 
                         [   (281, 64.722), 
                             (566, 61.306), 
                             (787, 78.237),
                             (243, 62.276),
                             (86,  60.460)     ], )
def test_CalcAreaPerMolecule(systems, systemid, result):
    from DatabankLib.databankLibrary import CalcAreaPerMolecule
    sys0 = systems.loc(systemid)
    apm = CalcAreaPerMolecule(sys0)
    assert abs(apm - result) < 6e-4

@pytest.mark.parametrize("systemid, result", 
                         [   (281, 4142.234), 
                             (566, 3923.568), 
                             (787, 4694.191),
                             (243, 2241.920),
                             (86,  3869.417)     ], )
def test_calcArea(systems, systemid, result):
    from DatabankLib.databankLibrary import calcArea
    sys0 = systems.loc(systemid)
    area = calcArea(sys0)
    assert abs(area - result) < 6e-4

@pytest.mark.parametrize("systemid, result", 
                         [   (281, 128), 
                             (566, 128), 
                             (787, 120),
                             (243, 72),
                             (86,  128)     ], )
def test_GetNLipids(systems, systemid, result):
    from DatabankLib.databankLibrary import GetNlipids
    sys0 = systems.loc(systemid)
    nlip = GetNlipids(sys0)
    assert nlip == result


@pytest.mark.parametrize("systemid, result", 
                         [   (281, [0.264, 0.415, 0.614]), 
                             (566, [0.263, 0.404, 0.604]), 
                             (243, [0.281, 0.423, 0.638]),
                             (86,  [0.266, 0.421, 0.63])     ], )
def test_GetFormFactorMin(systems, systemid, result):
    from DatabankLib.databankLibrary import GetFormFactorMin
    import numpy as np
    sys0 = systems.loc(systemid)
    ffl = GetFormFactorMin(sys0)
    err = ( (np.array(ffl[:3]) - np.array(result))**2 ).sum()
    assert err < 1e-9

@pytest.mark.parametrize("systemid, result", 
                         [   (281, 31.5625), 
                             (566, 31.0), 
                             (787, 75.0),
                             (243, 39.7778),
                             (86,  27.75)     ], )
def test_getHydrationLevel(systems, systemid, result):
    from DatabankLib.databankLibrary import getHydrationLevel
    sys0 = systems.loc(systemid)
    hl = getHydrationLevel(sys0)
    assert abs(hl - result) < 1e-4


@pytest.mark.parametrize("systemid, lipid, result", 
                         [   (281, ['POPC'], [1]), 
                             (566, ['POPC','CHOL'], [.9375,.0625]), 
                             (787, ['TOCL', 'POPC', 'POPE'], [0.25,0.5,0.25]),
                             (243, ['DPPC'], [1]),
                             (86,  ['POPE'], [1])     ], )
def test_calcLipidFraction(systems, systemid, lipid, result):
    from DatabankLib.databankLibrary import calcLipidFraction
    sys0 = systems.loc(systemid)
    assert calcLipidFraction(sys0, 'SOPC') == 0 # absent lipid
    err = 0
    i = 0
    for lip in lipid:
        err += ( calcLipidFraction(sys0, lip) - result[i] )**2
        i += 1
    assert err < 1e-4

@pytest.mark.parametrize("systemid, result", 
                         [   (281, [-0.1610, -0.1217] ), 
                             (566, [-0.1714, -0.1142]), 
                             (243, [-0.1764, -0.1784]),
                             (86,  [-0.1933, -0.1568])     ], )
def test_averageOrderParameters(systems, systemid, result):
    from DatabankLib.databankLibrary import averageOrderParameters
    sys0 = systems.loc(systemid)
    sn1,sn2 =  averageOrderParameters(sys0)
    assert (sn1-result[0])**2 + (sn2-result[1])**2 < 1e-5

## Tests behavior when averageOrderParameters cannot find calculated OP data

@pytest.mark.parametrize("systemid", [ 787], )
def test_raises_averageOrderParameters(systems, systemid):
    from DatabankLib.databankLibrary import averageOrderParameters
    sys0 = systems.loc(systemid)
    with pytest.raises(FileNotFoundError) as exc_info:
        sn1,sn2 =  averageOrderParameters(sys0)
    assert 'OrderParameters.json' in str(exc_info.value)


@pytest.mark.parametrize("systemid, lipid, result", 
                         [   (281, ['POPC/P'], ['M_G3P2_M']), 
                             (566, ['POPC/P31','CHOL/C1'], ['M_G3P2_M', 'M_C1_M']), 
                             (787, ['TOCL/P3', 'POPC/P', 'POPE/P'], ['M_G13P2_M', 'M_G3P2_M', 'M_G3P2_M']),
                             (243, ['DPPC/P8'], ['M_G3P2_M']),
                             (86,  ['POPE/P8'], ['M_G3P2_M'])     ], )
def test_getUniversalAtomName(systems, systemid, lipid, result):
    from DatabankLib.databankLibrary import getUniversalAtomName
    sys0 = systems.loc(systemid)
    i = 0
    for lipat in lipid:
        lip,atom = tuple(lipat.split('/'))
        uname = getUniversalAtomName(sys0, atom, lip)
        assert uname == result[i]
        i += 1

## Test fail-behavior of getUniversalAtomName

@pytest.mark.parametrize("systemid, lipat, result", 
                         [   (243, 'DPPC/nonExisting', "Atom was not found"),
                             (243, 'nonExisting/P8', "Mapping file was not found") ] )
def test_bad_getUniversalAtomName(systems, systemid, lipat, result, capsys):
    from DatabankLib.databankLibrary import getUniversalAtomName
    sys0 = systems.loc(systemid)
    lip,atom = tuple(lipat.split('/'))
    uname = getUniversalAtomName(sys0, atom, lip)
    output = capsys.readouterr().err.rstrip()
    assert result in output
    assert uname is None

@pytest.mark.parametrize("systemid, lipid, result", 
                         [   (243, 'DPPC', "44ea5"),
                             (787, 'TOCL', "78629") ] )
def test_getAtoms(systems, systemid, lipid, result):
    from DatabankLib.databankLibrary import getAtoms
    sys0 = systems.loc(systemid)
    atoms = getAtoms(sys0, lipid).split()
    atoms = ",".join(sorted(atoms))
    import hashlib
    md5_hash = hashlib.md5()
    md5_hash.update(atoms.encode('ascii'))
    hx = md5_hash.hexdigest()[:5]
    assert hx == result

@pytest.mark.parametrize("systemid, lipid, result", 
                         [   (281, ['POPC'], [134]), 
                             (566, ['POPC','CHOL'], [134, 74]), 
                             (787, ['TOCL', 'POPC', 'POPE'], [248, 134, 125]),
                             (243, ['DPPC'], [130]),
                             (86,  ['POPE'], [125])     ], )
def test_loadMappingFile(systems, systemid, lipid, result):
    from DatabankLib.databankLibrary import loadMappingFile
    sys0 = systems.loc(systemid)
    i = 0
    for lip in lipid:
        mpf = loadMappingFile(sys0['COMPOSITION'][lip]['MAPPING'])
        assert len(mpf) == result[i]
        i += 1

@pytest.mark.xfail(reason="Improper file name", run=True, 
                   raises=FileNotFoundError, strict=True)
def test_raise_loadMappingFile():
    from DatabankLib.databankLibrary import loadMappingFile
    mpf = loadMappingFile('file-doesnot-exist')
    print(mpf)


@pytest.mark.parametrize("systemid, lipid, result", 
                         [   (281, ['POPC/P'], ['M_G3P2_M']), 
                             (566, ['POPC/P31','CHOL/C1'], ['M_G3P2_M', 'M_C1_M']), 
                             (787, ['TOCL/P3', 'POPC/P', 'POPE/P'], ['M_G13P2_M', 'M_G3P2_M', 'M_G3P2_M']),
                             (243, ['DPPC/P8'], ['M_G3P2_M']),
                             (86,  ['POPE/P8'], ['M_G3P2_M'])     ], )
def test_simulation2universal_atomnames(systems, systemid, lipid, result):
    from DatabankLib.databankLibrary import simulation2universal_atomnames
    sys0 = systems.loc(systemid)
    i = 0
    for lipat in lipid:
        lip,atom = tuple(lipat.split('/'))
        sname = simulation2universal_atomnames(sys0, lip, result[i])
        assert sname == atom
        i += 1


@pytest.mark.parametrize("systemid, lipat, result", 
                         [   (243, 'DPPC/nonExisting', "was not found from mappingDPPCberger.yaml"),
                             (243, 'nonExisting/M_G1_M', "Mapping file was not found") ] )
def test_bad_simulation2universal_atomnames(systems, systemid, lipat, result, capsys):
    from DatabankLib.databankLibrary import simulation2universal_atomnames
    sys0 = systems.loc(systemid)
    lip,atom = tuple(lipat.split('/'))
    sname = simulation2universal_atomnames(sys0, lip, atom)
    output = capsys.readouterr().err.rstrip()
    assert result in output
    assert sname is None


@pytest.mark.parametrize("systemid, result", 
                         [   (281, "resname POPC"), 
                             (566, "resname CHL or resname OL or resname PA or resname PC"), 
                             (787, "resname POPC or resname POPE or resname TOCL2"),
                             (243, "resname DPPC"),
                             (86,  "resname POPE")     ], )
def test_getLipids(systems, systemid, result):
    from DatabankLib.databankLibrary import getLipids
    sys0 = systems.loc(systemid)
    gl = getLipids(sys0)
    assert gl == result

## TEST thickness calculation here because it is not trajectory-based, but JSON-based

@pytest.mark.parametrize("systemid, result", 
                         [   (566, DatabankLib.RCODE_SKIPPED), 
                             (787, DatabankLib.RCODE_ERROR),
                             (86,  DatabankLib.RCODE_SKIPPED)     ], )
def test_analyze_th(systems, systemid, result):
    from DatabankLib.analyze import computeTH
    sys0 = systems.loc(systemid)
    rc = computeTH(sys0)
    assert rc == result
    if rc == DatabankLib.RCODE_ERROR:
        fn = os.path.join(DatabankLib.NMLDB_SIMU_PATH, 
                    sys0['path'],
                    'thickness.json')
        assert os.path.isfile(fn) # file is not created

@pytest.fixture(scope='function')
def wipeth(systems):
    # TD-FIXTURE FOR REMOVING THICKNESS AFTER TEST CALCULATIONS
    yield
    # TEARDOWN
    for sid in [243,281]:
        sys0 = systems.loc(sid)
        fn = os.path.join(DatabankLib.NMLDB_SIMU_PATH, sys0['path'], 'thickness.json')
        try:
            os.remove(fn)
        except:
            pass

@pytest.mark.parametrize("systemid, result, thickres", 
                         [   (281, DatabankLib.RCODE_COMPUTED, 4.18335), 
                             (243, DatabankLib.RCODE_COMPUTED, 4.27262) ], )
def test_analyze_th(systems, systemid, result, wipeth, thickres):
    from DatabankLib.analyze import computeTH
    sys0 = systems.loc(systemid)
    rc = computeTH(sys0)
    assert rc == result
    fn = os.path.join(DatabankLib.NMLDB_SIMU_PATH, 
                    sys0['path'],
                    'thickness.json')
    assert os.path.isfile(fn)
    with open(fn, 'r') as file:
        data = float(file.read().rstrip())
    assert abs(data - thickres) < 1e-5
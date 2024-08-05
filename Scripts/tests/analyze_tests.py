from unittest import mock
import os, glob
import pytest
import DatabankLib

@pytest.fixture(autouse=True, scope="session")
def header_session_scope():
    _rp = os.path.join(os.path.dirname(__file__), "Data", "Simulations")
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
    ## TEARDOWN SYSTEMS
    for _s in s:
        for f in glob.glob(os.path.join(DatabankLib.NMLDB_SIMU_PATH, _s['path'], '*.json')):
            os.remove(f)


@pytest.fixture(scope="module")
def systemLoadTraj(systems):
    from DatabankLib.databankLibrary import system2MDanalysisUniverse
    print('DBG: Download trajectory data.')
    for s in systems:
        u = system2MDanalysisUniverse(s)
    yield
    ## TEARDOWN SYSTEM-LOADING
    print('DBG: Wiping trajectory data.')
    for s in systems:
        files_ = [ s['GRO'][0][0], s['TPR'][0][0], s['TRJ'][0][0] ]
        files_ = map(lambda x: os.path.join(DatabankLib.NMLDB_SIMU_PATH, s['path'], x), files_)
        for f in files_:
            print(f)
            if os.path.exists(f):
                os.remove(f)


def test_analyze_apl(systems, systemLoadTraj):
    from DatabankLib.analyze import computeAPL
    aplCreated = []
    aplNZ = []
    for s in systems:
        computeAPL(s)
        cFile = os.path.join(DatabankLib.NMLDB_SIMU_PATH, 
                                        s['path'], 'apl.json')
        aplCreated.append( os.path.isfile(cFile) )
        aplNZ.append( os.path.getsize(cFile) > 1e3 )
    assert all(aplCreated) and all(aplNZ)
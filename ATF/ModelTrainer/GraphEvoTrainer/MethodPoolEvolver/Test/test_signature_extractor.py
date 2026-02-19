# test_signature_extractor.py
import pytest
import networkx as nx
from adaptoflux.model_trainer.method_pool_evolver.signature_extractor import SignatureExtractor

class MockTrainer:
    def __init__(self, verbose=True):
        self.verbose = verbose

@pytest.fixture
def extractor():
    return SignatureExtractor(MockTrainer())

@pytest.fixture
def sample_graph():
    g = nx.DiGraph()
    g.add_node("0_0_add", method_name="add", layer=0)
    g.add_node("1_0_relu", method_name="relu", layer=1)
    g.add_edge("0_0_add", "1_0_relu", data_coord=0, data_type='scalar')
    g.add_node("root")
    g.add_node("collapse")
    return g

def test_extract_signature_normal_node(extractor, sample_graph):
    sig = extractor.extract_signature(sample_graph, "0_0_add")
    assert sig[0] == 0  # layer
    assert sig[1] == ()  # no in-edges
    assert sig[2] == (0,)  # one out-edge with coord=0

def test_extract_signature_special_node_raises(extractor, sample_graph):
    with pytest.raises(ValueError):
        extractor.extract_signature(sample_graph, "root")

def test_build_frequency_map_basic(extractor, sample_graph):
    class MockSnap:
        def __init__(self, g):
            class GP: 
                graph = g
            self.graph_processor = GP()
    
    snapshots = [MockSnap(sample_graph), MockSnap(sample_graph)]
    freq_map = extractor.build_frequency_map(snapshots)
    
    sig = (0, (), (0,))
    assert freq_map[sig]["add"] == 2  # appeared in 2 snapshots
from workflow_utils import visualization
import seaborn as sns


def test_histograms():
    data = sns.load_dataset('iris')
    _ = visualization.histograms(data)
    assert True

def test_kdes():
    data = sns.load_dataset('iris')
    _ = visualization.kdes(data)
    assert True
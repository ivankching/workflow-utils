import seaborn as sns
from workflow_utils import stats


def test_vif():
    data = stats.vif(sns.load_dataset('iris'), exclude=['species'])
    assert data.shape == (4,2)
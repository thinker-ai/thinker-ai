import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import unittest

from thinker_ai.machine_learning.data_analyzer import DataAnalyzer


def create_hot_topic_data() -> pd.DataFrame:
    np.random.seed(0)
    periods = 300
    dates = pd.date_range('2021-01-01', periods=periods, freq='D')
    num_topics = 5
    data = pd.DataFrame(np.random.randn(periods, num_topics), index=dates,
                        columns=[f'Hot_Topic_{i}' for i in range(1, num_topics + 1)])
    # Introduce some known correlations
    data['Hot_Topic_3'] = data['Hot_Topic_1'] * 0.5  # Positive correlation with Topic 1
    data['Hot_Topic_4'] = -data['Hot_Topic_1'] * 0.5  # Negative correlation with Topic 1
    return data


def plot_correlation_heatmap(corr_matrix):
    plt.figure(figsize=(10, 8))  # Adjust the figure size as necessary
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.tight_layout()  # This will fit the plot within the figure area without clipping
    plt.show()


# The unit test class
class TestCorrelationAnalysis(unittest.TestCase):

    def test_calculate_correlations(self):
        data = create_hot_topic_data()
        corr_matrix = DataAnalyzer.correlations(data)
        plot_correlation_heatmap(corr_matrix)
        # Test for positive correlation
        self.assertGreater(corr_matrix.loc['Hot_Topic_1', 'Hot_Topic_3'], 0.4)
        # Test for negative correlation
        self.assertLess(corr_matrix.loc['Hot_Topic_1', 'Hot_Topic_4'], -0.4)


# Run the unit tests
if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)

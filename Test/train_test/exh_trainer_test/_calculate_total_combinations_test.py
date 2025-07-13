import unittest
from ATF.ModelTrainer.ExhTrainer.exh_trainer import ExhaustiveSearchEngine


class TestCalculateTotalCombinations(unittest.TestCase):

    def test_calculate_total_combinations(self):
        class MockAdaptoflux:
            def __init__(self):
                self.feature_types = ['numerical', 'numerical']
                self.values = None  # 可以忽略，因为只用到了 feature_types

            def build_function_pool_by_input_type(self, _):
                return {
                    'numerical': [
                        ('f1', {'output_types': ['numerical']}),
                        ('f2', {'output_types': ['text']})
                    ]
                }


        # 初始化引擎
        engine = ExhaustiveSearchEngine(MockAdaptoflux())

        output_sizes = [len(engine.adaptoflux.feature_types)]  # 初始输入维度
        num_layers = 2

        total_combinations, updated_output_sizes = engine._calculate_total_combinations(
            num_layers, output_sizes
        )

        # 验证组合数是否正确
        self.assertEqual(total_combinations, 9)
        self.assertEqual(updated_output_sizes, [2, 4, 9])


if __name__ == '__main__':
    unittest.main()
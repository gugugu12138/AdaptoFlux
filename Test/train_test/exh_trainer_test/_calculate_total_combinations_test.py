import unittest
from ATF.ModelTrainer.ExhTrainer.exh_trainer import ExhaustiveSearchEngine


class TestCalculateTotalCombinations(unittest.TestCase):

    def test_calculate_total_combinations(self):
        class MockAdaptoflux:
            def __init__(self):
                self.feature_types = ['numerical', 'numerical','one']
                self.values = None  # 可以忽略，因为只用到了 feature_types
                # 手动构造 methods 字段
                self.methods = {
                    'f1': {
                        "input_count": 1,
                        "output_count": 1,
                        "input_types": ["numerical"],
                        "output_types": ["numerical"],
                        "group": "transform",
                        "weight": 1.0,
                        "vectorized": True,
                        "function": lambda x: x  # 可选：添加一个 dummy 函数
                    },
                    'f2': {
                        "input_count": 1,
                        "output_count": 1,
                        "input_types": ["numerical"],
                        "output_types": ["text"],
                        "group": "transform",
                        "weight": 1.0,
                        "vectorized": False,
                        "function": lambda x: x
                    },
                    'f3': {
                        "input_count": 1,
                        "output_count": 1,
                        "input_types": ["one"],
                        "output_types": ["numerical"],
                        "group": "transform",
                        "weight": 1.0,
                        "vectorized": True,
                        "function": lambda x: x  # 可选：添加一个 dummy 函数
                    },
                    '__empty__': {  # 可选：支持空函数
                        "input_count": 1,
                        "output_count": 1,
                        "input_types": ["None"],
                        "output_types": ["None"],
                        "group": "none",
                        "weight": 0.0,
                        "vectorized": True,
                        "function": lambda x: x
                    }
                }

            def build_function_pool_by_input_type(self):
                return {
                    'numerical': [
                        ('f1', {
                            "input_count": 1,
                            "output_count": 1,
                            "input_types": ["numerical"],
                            "output_types": ["numerical"],
                            "group": "transform",
                            "weight": 1.0,
                            "vectorized": True
                        }),
                        ('f2', {
                            "input_count": 1,
                            "output_count": 1,
                            "input_types": ["numerical"],
                            "output_types": ["text"],
                            "group": "transform",
                            "weight": 1.0,
                            "vectorized": False
                        })
                    ],
                    'one': [
                        ('f3', {
                            "input_count": 1,
                            "output_count": 1,
                            "input_types": ["one"],
                            "output_types": ["numerical"],
                            "group": "transform",
                            "weight": 1.0,
                            "vectorized": True
                        })]
                }


        # 初始化引擎
        engine = ExhaustiveSearchEngine(MockAdaptoflux())

        num_layers = 1

        total_combinations, tree = engine._calculate_total_combinations(
            num_layers
        )

        # 验证组合数是否正确
        self.assertEqual(total_combinations, )


if __name__ == '__main__':
    unittest.main()
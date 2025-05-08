import random
from copy import deepcopy

class TestProcessor:
    def __init__(self):
        self.methods = {
            "method_A": {"input_count": 2, "output_count": 1},
            "method_B": {"input_count": 3, "output_count": 1}
        }

    def group_indices_by_input_count(self, indices, input_count, shuffle=False):
        if shuffle:
            random.shuffle(indices)
        groups = []
        unmatched = []
        i = 0
        while i < len(indices):
            group = indices[i:i + input_count]
            if len(group) == input_count:
                groups.append(group)
            else:
                unmatched.extend([idx] for idx in group)
            i += input_count
        return groups, unmatched

    def process_random_method(self, shuffle_indices=True):
        """
        模拟生成初始 method_group_positions + index_map
        """
        if not self.methods:
            raise ValueError("方法字典为空，无法处理！")

        num_elements = 10
        method_list = []

        # 初始分配方法名
        for _ in range(num_elements):
            method_name = random.choice(list(self.methods.keys()))
            method_list.append(method_name)

        multi_input_positions = {}
        for idx, method_name in enumerate(method_list):
            multi_input_positions.setdefault(method_name, []).append(idx)

        index_map = {}
        valid_groups = {}
        unmatched = []

        for method_name, indices in multi_input_positions.items():
            input_count = self.methods[method_name]["input_count"]
            groups, unmatch = self.group_indices_by_input_count(indices, input_count, shuffle=shuffle_indices)

            valid_groups[method_name] = groups
            for group in groups:
                for idx in group:
                    index_map[idx] = {"method": method_name, "group": tuple(group)}
            # 未分组的数据每一个单独视为一个组
            unmatched.extend(idx for idx in unmatch)

        # 新增：将 unmatched 中的索引也加入 index_map
        for sublist in unmatched:
            for idx in sublist:
                print(idx)
                index_map[idx] = {
                    "method": "unmatched",
                    "group": tuple([idx])  # 每个 idx 自己成为一个 group
                }

        return {
            "index_map": index_map,
            "valid_groups": valid_groups,
            "unmatched": unmatched
        }
    
    def replace_random_elements(self, result, n, shuffle_indices=True):
        """
        替换部分索引对应的方法，并重新分组以符合 input_count 要求。
        
        :param result: 来自 process_random_method 的输出结构
        :param n: 要随机替换的索引数量
        :param shuffle_indices: 是否打乱新方法分配时的索引顺序
        :return: 新的 result 结构，包含更新后的 index_map、valid_groups 和 unmatched
        """
        from copy import deepcopy

        # 提取原始信息
        index_map = deepcopy(result["index_map"])
        valid_groups = deepcopy(result["valid_groups"])
        original_unmatched = deepcopy(result.get("unmatched", []))

        # 新增集合：记录所有将被替换的索引（包括同 group 成员）
        indices_to_be_assigned = set()

        # 收集所有可替换的索引条目 (method_name, idx)
        all_index_entries = []
        for idx in index_map:
            info = index_map[idx]
            method_name = info["method"]
            all_index_entries.append((method_name, idx))  # 只保留 method_name 和 idx

        total_indices = len(all_index_entries)
        if n > total_indices:
            raise ValueError(f"无法替换 {n} 个索引，总数只有 {total_indices}")

        # 随机选取 n 个要替换的索引
        indices_to_replace = random.sample(all_index_entries, n)

        # 深拷贝用于修改的数据结构
        new_valid_groups = deepcopy(valid_groups)
        new_unmatched = deepcopy(original_unmatched)
        new_index_map = deepcopy(index_map)

        # 删除这些索引所属的 group（使用 index_map 获取信息）
        for method_name, idx in indices_to_replace:
            info = new_index_map.get(idx)
            if not info:
                continue  # 如果已经被删了就跳过

            current_method = info["method"]
            current_group = info["group"]

            # 处理 valid_groups 中的方法
            if current_method != "unmatched":
                if current_method in new_valid_groups:
                    for i, group in enumerate(new_valid_groups[current_method]):
                        if tuple(group) == current_group:
                            del new_valid_groups[current_method][i]
                            break

                # 找出这个 group 中的所有索引，并全部加入待替换集合
                for other_idx in list(new_index_map.keys()):
                    other_info = new_index_map.get(other_idx)
                    if other_info and other_info["group"] == current_group:
                        indices_to_be_assigned.add(other_idx)

            # 处理 unmatched 中的索引
            else:
                for i, group_candidate in enumerate(new_unmatched):
                    if idx in group_candidate:
                        del new_unmatched[i]
                        break
                # 将整个 group_candidate 中的索引都加入待替换集合
                for other_idx in group_candidate:
                    indices_to_be_assigned.add(other_idx)

        # 从 new_index_map 中删除所有待替换的索引
        for idx in indices_to_be_assigned:
            if idx in new_index_map:
                del new_index_map[idx]

        # 转换为列表用于后续操作
        indices_to_be_assigned = list(indices_to_be_assigned)

        # 分配新方法给这些索引
        new_assignments = {}
        for idx in indices_to_be_assigned:
            method_name = random.choice(list(self.methods.keys()))
            new_assignments.setdefault(method_name, []).append(idx)

        # 按照 input_count 分组
        assignments_method_group_positions = {"unmatched": []}
        for method_name, indices in new_assignments.items():
            input_count = self.methods[method_name]["input_count"]
            if shuffle_indices:
                random.shuffle(indices)
            groups, unmatch = self.group_indices_by_input_count(indices, input_count, shuffle=shuffle_indices)
            assignments_method_group_positions[method_name] = groups
            assignments_method_group_positions["unmatched"].extend([[idx] for idx in unmatch])

        # 合并老的和新的 valid_groups
        updated_valid_groups = deepcopy(new_valid_groups)
        for method_name, groups in assignments_method_group_positions.items():
            if not groups or method_name == "unmatched":
                continue
            if method_name in updated_valid_groups:
                updated_valid_groups[method_name].extend(groups)
            else:
                updated_valid_groups[method_name] = groups

        # 构建新的 index_map（只更新被替换的部分 + 新增部分）
        for method_name, groups in assignments_method_group_positions.items():
            if method_name == "unmatched":
                continue
            for group in groups:
                for idx in group:
                    new_index_map[idx] = {
                        "method": method_name,
                        "group": tuple(group)
                    }

        # 处理 unmatched 中的索引：将每个 idx 都作为独立 group 加入 index_map
        for group_candidate in assignments_method_group_positions.get("unmatched", []):
            # group_candidate 可能是 [2], [3, 4], [5, 6, 7] 等
            for sublist in group_candidate:
                for idx in sublist:
                    new_index_map[idx] = {
                        "method": "unmatched",
                        "group": tuple([idx])  # 每个 idx 自己成为一个 group
                    }

        # 处理未匹配项（unmatched）← 保留原来的所有 + 新加入的
        final_unmatched = deepcopy(new_unmatched)
        final_unmatched.extend(assignments_method_group_positions["unmatched"])

        return {
            "index_map": new_index_map,
            "valid_groups": updated_valid_groups,
            "unmatched": final_unmatched
        }

def test_replace_functions():
    processor = TestProcessor()

    # Step 1: 生成初始数据
    process_result = processor.process_random_method(shuffle_indices=True)

    print("原始 process_result:")
    print(process_result)

    # Step 2: 调用第一个函数（适配新结构）
    new_result_1 = processor.replace_random_elements(process_result, n=4, shuffle_indices=True)
    print("\n【第一个函数 replace_random_elements】输出:")
    print(new_result_1)

if __name__ == "__main__":
    test_replace_functions()

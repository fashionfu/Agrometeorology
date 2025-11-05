#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
提取0级预警规则
"""

import os
import re
from pathlib import Path


def parse_rule_file(file_path: str):
    """解析规则文件"""
    if not os.path.exists(file_path):
        print(f"警告: 文件不存在: {file_path}")
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"错误: 无法读取文件 {file_path}: {e}")
        return []
    
    rules = []
    current_rule = None
    current_condition = []
    
    for line in lines:
        line = line.strip()
        
        # 匹配规则开头: 规则1: 覆盖样本=X, 准确率=Y
        rule_match = re.match(r'规则(\d+):\s*覆盖样本=(\d+),\s*准确率=([\d.]+)', line)
        if rule_match:
            # 保存之前的规则
            if current_rule is not None:
                rules.append(current_rule)
            
            # 开始新规则
            current_rule = {
                'rule_num': rule_match.group(1),
                'samples': int(rule_match.group(2)),
                'accuracy': float(rule_match.group(3)),
                'condition': '',
                'class_dist': '',
                'score': 0
            }
            current_condition = []
            continue
        
        # 匹配条件行
        if line.startswith('条件:') or line.startswith('条件：'):
            condition_part = line.split(':', 1)[-1].strip()
            if condition_part:
                current_condition.append(condition_part)
            continue
        
        # 如果当前有规则且在收集条件
        if current_rule is not None and current_condition:
            if line and not line.startswith('类别分布') and not line.startswith('评分'):
                current_condition.append(line)
            elif line.startswith('类别分布'):
                # 条件收集完成
                current_rule['condition'] = ' 且 '.join(current_condition).strip()
                current_condition = []
                
                # 提取类别分布
                class_match = re.search(r'\{([^}]+)\}', line)
                if class_match:
                    current_rule['class_dist'] = class_match.group(1)
                continue
        
        # 匹配类别分布（如果在单独的行）
        if current_rule is not None and '类别分布' in line:
            class_match = re.search(r'\{([^}]+)\}', line)
            if class_match:
                current_rule['class_dist'] = class_match.group(1)
            continue
    
    # 保存最后一个规则
    if current_rule is not None:
        if current_condition:
            current_rule['condition'] = ' 且 '.join(current_condition).strip()
        current_rule['score'] = current_rule['samples'] * current_rule['accuracy']
        rules.append(current_rule)
    
    # 按评分排序
    rules.sort(key=lambda x: x['score'], reverse=True)
    return rules


def extract_level0_rules():
    """提取三个方案的0级预警规则"""
    base_dir = Path("analysis_1104_batch")
    
    schemes = [
        ("方案一：最佳准确率方案 (depth4_leaf2)", "depth4_leaf2", base_dir / "analysis_depth4_leaf2" / "tree_rules_warning_0.txt"),
        ("方案二：平衡方案 (depth4_leaf3)", "depth4_leaf3", base_dir / "analysis_depth4_leaf3" / "tree_rules_warning_0.txt"),
        ("方案三：特征丰富度方案 (depth7_leaf2)", "depth7_leaf2", base_dir / "analysis_depth7_leaf2" / "tree_rules_warning_0.txt"),
    ]
    
    all_results = {}
    
    for scheme_name, scheme_id, file_path in schemes:
        print(f"\n正在处理: {scheme_name}")
        print(f"文件路径: {file_path}")
        
        if file_path.exists():
            rules = parse_rule_file(str(file_path))
            all_results[scheme_id] = {
                'name': scheme_name,
                'rules': rules
            }
            print(f"✓ 找到 {len(rules)} 条规则")
            for i, rule in enumerate(rules, 1):
                print(f"  规则{i}: 覆盖样本={rule['samples']}, 准确率={rule['accuracy']:.3f}, score={rule['score']:.2f}")
        else:
            print(f"✗ 文件不存在")
            all_results[scheme_id] = {
                'name': scheme_name,
                'rules': []
            }
    
    return all_results


def format_rules_output(results):
    """格式化输出规则，按照用户要求的格式"""
    output = []
    output.append("# 0级（不发生）预警规则\n")
    
    for scheme_id, data in results.items():
        scheme_name = data['name']
        rules = data['rules']
        
        output.append(f"## {scheme_name}\n")
        
        if not rules:
            output.append("*注：未找到规则或文件不存在*\n\n")
            continue
        
        for rule in rules:
            accuracy_percent = int(rule['accuracy'] * 100)
            output.append(f"**规则{rule['rule_num']}（覆盖样本={rule['samples']}，准确率={accuracy_percent}%）**：\n\n")
            output.append(f"条件: {rule['condition']}\n\n")
            if rule['class_dist']:
                output.append(f"类别分布: {{{rule['class_dist']}}}\n\n")
            output.append(f"评分: score={rule['score']:.0f}={rule['samples']}*{rule['accuracy']:.2f}\n\n")
        
        # 最优规则
        if rules:
            best_rule = rules[0]
            output.append(f"**最优规则**: {best_rule['condition']}（samples={best_rule['samples']}，准确率={best_rule['accuracy']:.3f}，score={best_rule['score']:.0f}）\n\n")
    
    return "\n".join(output)


if __name__ == "__main__":
    print("="*60)
    print("开始提取0级预警规则")
    print("="*60)
    
    results = extract_level0_rules()
    output = format_rules_output(results)
    
    # 保存到文件
    output_file = "0级预警规则提取.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output)
    
    print(f"\n{'='*60}")
    print(f"✓ 结果已保存到: {output_file}")
    print("="*60)
    
    # 同时打印到控制台
    print("\n提取的规则内容:\n")
    print(output)

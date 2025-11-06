#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查数据文件格式"""
import pandas as pd

df = pd.read_excel('样本数据.xlsx', sheet_name=0)
print('列名:', list(df.columns))
print('行数:', len(df))
print('\n前5行:')
print(df.head())
print('\n预警列唯一值:', df['预警'].unique() if '预警' in df.columns else '未找到预警列')


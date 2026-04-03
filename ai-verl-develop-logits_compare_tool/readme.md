# Logits Comparison Tool

## 功能
- 批量推理，支持多GPU并行加速
- 对比原生模型、训练后模型token维度的logp
- 计算联合概率(joint logp)
- 结果分析与可视化

## 使用方法

### 1. 跑生成
```bash
bash run_compare.sh
```
```bash
bash run_compare.sh \
    --base_model /path/to/base/model \
    --trained_model /path/to/trained/model \
    --data_path /path/to/test_data.json \
    --output_path /path/to/results.json \
    --batch_size 16 \
    --device_ids "0,1,2,3"
```

### 2. 结果分析
```bash
python visualize_results_v2.py --results_path /path/to/results.json --output_dir /path/to/output
```


## 输出格式
生成结果JSON包含每个样本的：
`base_joint_logp`: 原生模型的联合对数概率
`trained_joint_logp`: 训练后模型的联合对数概率
`joint_logp_diff`: 联合对数概率差异
`base_token_logps`: 原生模型每个token的对数概率列表
`trained_token_logps`: 训练后模型每个token的对数概率列表
`token_logp_diffs`: 每个token的对数概率差异列表

分析结果JSON实例说明：
```json
  "sample_level": { // 样本维度
    "total_samples": 992,
    "joint_logp_diff": {
      ...
      "improved_samples": 933,
      "degraded_samples": 59
    }
  },
  "token_level": { // token维度
    "all_tokens": {
      "total_tokens": 105393,
      "logp_diff": {
        ...
        "improved_tokens": 38609,
        "degraded_tokens": 56761
      }
    },
    "filtered_tokens_abs_diff_gte_0.1": {  // 加筛选条件
      "total_tokens": 13672,
      "logp_diff": {
        ...
        "improved_tokens": 12081,
        "degraded_tokens": 1591
      }
    }
  },
  "position_level_analysis": { // token索引
    "top_decrease_positions": [...],
    "top_increase_positions":[...]
  }
```
## 性能优化
- 使用批处理加速推理
- 支持DataParallel多GPU并行
- FP16精度降低显存占用

## TODO
- acclerate性能加速，提升GPU利用率


# Useful tools for open_clip

> 专为[open_clip](https://github.com/mlfoundations/open_clip)设计的工具集

* [__torch_mha_2_naive_mha__](./torch_mha_2_naive_mha/README.md)：将pytorch多头注意力转换为QKV分离的注意力实现，方便对QKV投影分别操作。
* [__peft_lora__](./peft_lora/README.md)：向QKV分离的注意力模块插入LoRA适配器。
* [__visualize__](./visualize/README.md)：给定文本描述，可视化CLIP高激活区域。

## Usage

本工具集已在`open_clip_torch==3.3.0`、`transformers==5.4.0`和`torch==2.6.0`环境下验证正确性。

```bash
pip install -r requirements.txt
```

随后参考每个子工具下的`__main__`或`example.py`。
import pybuda
import pybuda.op
from pybuda import PyBudaModule

from loguru import logger
import torch


class AslModel(PyBudaModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter("vit.embeddings.cls_token", pybuda.Parameter(*(1, 1, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.embeddings.patch_embeddings.projection.weight", pybuda.Parameter(*(768, 3, 16, 16), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.embeddings.patch_embeddings.projection.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.embeddings.position_embeddings", pybuda.Parameter(*(1, 197, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.0.layernorm_before.weight", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.0.layernorm_before.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.0.attention.attention.query.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.0.attention.attention.query.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.0.attention.attention.key.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.0.attention.attention.key.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.0.attention.attention.value.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.0.attention.attention.value.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.0.attention.output.dense.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.0.attention.output.dense.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.0.layernorm_after.weight", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.0.layernorm_after.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.0.intermediate.dense.weight", pybuda.Parameter(*(3072, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.0.intermediate.dense.bias", pybuda.Parameter(*(3072,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.0.output.dense.weight", pybuda.Parameter(*(768, 3072), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.0.output.dense.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.1.layernorm_before.weight", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.1.layernorm_before.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.1.attention.attention.query.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.1.attention.attention.query.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.1.attention.attention.key.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.1.attention.attention.key.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.1.attention.attention.value.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.1.attention.attention.value.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.1.attention.output.dense.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.1.attention.output.dense.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.1.layernorm_after.weight", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.1.layernorm_after.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.1.intermediate.dense.weight", pybuda.Parameter(*(3072, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.1.intermediate.dense.bias", pybuda.Parameter(*(3072,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.1.output.dense.weight", pybuda.Parameter(*(768, 3072), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.1.output.dense.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.2.layernorm_before.weight", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.2.layernorm_before.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.2.attention.attention.query.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.2.attention.attention.query.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.2.attention.attention.key.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.2.attention.attention.key.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.2.attention.attention.value.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.2.attention.attention.value.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.2.attention.output.dense.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.2.attention.output.dense.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.2.layernorm_after.weight", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.2.layernorm_after.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.2.intermediate.dense.weight", pybuda.Parameter(*(3072, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.2.intermediate.dense.bias", pybuda.Parameter(*(3072,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.2.output.dense.weight", pybuda.Parameter(*(768, 3072), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.2.output.dense.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.3.layernorm_before.weight", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.3.layernorm_before.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.3.attention.attention.query.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.3.attention.attention.query.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.3.attention.attention.key.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.3.attention.attention.key.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.3.attention.attention.value.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.3.attention.attention.value.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.3.attention.output.dense.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.3.attention.output.dense.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.3.layernorm_after.weight", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.3.layernorm_after.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.3.intermediate.dense.weight", pybuda.Parameter(*(3072, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.3.intermediate.dense.bias", pybuda.Parameter(*(3072,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.3.output.dense.weight", pybuda.Parameter(*(768, 3072), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.3.output.dense.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.4.layernorm_before.weight", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.4.layernorm_before.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.4.attention.attention.query.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.4.attention.attention.query.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.4.attention.attention.key.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.4.attention.attention.key.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.4.attention.attention.value.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.4.attention.attention.value.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.4.attention.output.dense.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.4.attention.output.dense.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.4.layernorm_after.weight", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.4.layernorm_after.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.4.intermediate.dense.weight", pybuda.Parameter(*(3072, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.4.intermediate.dense.bias", pybuda.Parameter(*(3072,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.4.output.dense.weight", pybuda.Parameter(*(768, 3072), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.4.output.dense.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.5.layernorm_before.weight", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.5.layernorm_before.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.5.attention.attention.query.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.5.attention.attention.query.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.5.attention.attention.key.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.5.attention.attention.key.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.5.attention.attention.value.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.5.attention.attention.value.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.5.attention.output.dense.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.5.attention.output.dense.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.5.layernorm_after.weight", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.5.layernorm_after.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.5.intermediate.dense.weight", pybuda.Parameter(*(3072, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.5.intermediate.dense.bias", pybuda.Parameter(*(3072,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.5.output.dense.weight", pybuda.Parameter(*(768, 3072), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.5.output.dense.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.6.layernorm_before.weight", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.6.layernorm_before.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.6.attention.attention.query.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.6.attention.attention.query.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.6.attention.attention.key.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.6.attention.attention.key.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.6.attention.attention.value.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.6.attention.attention.value.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.6.attention.output.dense.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.6.attention.output.dense.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.6.layernorm_after.weight", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.6.layernorm_after.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.6.intermediate.dense.weight", pybuda.Parameter(*(3072, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.6.intermediate.dense.bias", pybuda.Parameter(*(3072,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.6.output.dense.weight", pybuda.Parameter(*(768, 3072), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.6.output.dense.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.7.layernorm_before.weight", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.7.layernorm_before.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.7.attention.attention.query.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.7.attention.attention.query.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.7.attention.attention.key.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.7.attention.attention.key.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.7.attention.attention.value.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.7.attention.attention.value.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.7.attention.output.dense.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.7.attention.output.dense.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.7.layernorm_after.weight", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.7.layernorm_after.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.7.intermediate.dense.weight", pybuda.Parameter(*(3072, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.7.intermediate.dense.bias", pybuda.Parameter(*(3072,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.7.output.dense.weight", pybuda.Parameter(*(768, 3072), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.7.output.dense.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.8.layernorm_before.weight", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.8.layernorm_before.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.8.attention.attention.query.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.8.attention.attention.query.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.8.attention.attention.key.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.8.attention.attention.key.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.8.attention.attention.value.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.8.attention.attention.value.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.8.attention.output.dense.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.8.attention.output.dense.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.8.layernorm_after.weight", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.8.layernorm_after.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.8.intermediate.dense.weight", pybuda.Parameter(*(3072, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.8.intermediate.dense.bias", pybuda.Parameter(*(3072,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.8.output.dense.weight", pybuda.Parameter(*(768, 3072), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.8.output.dense.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.9.layernorm_before.weight", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.9.layernorm_before.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.9.attention.attention.query.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.9.attention.attention.query.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.9.attention.attention.key.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.9.attention.attention.key.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.9.attention.attention.value.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.9.attention.attention.value.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.9.attention.output.dense.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.9.attention.output.dense.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.9.layernorm_after.weight", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.9.layernorm_after.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.9.intermediate.dense.weight", pybuda.Parameter(*(3072, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.9.intermediate.dense.bias", pybuda.Parameter(*(3072,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.9.output.dense.weight", pybuda.Parameter(*(768, 3072), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.9.output.dense.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.10.layernorm_before.weight", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.10.layernorm_before.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.10.attention.attention.query.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.10.attention.attention.query.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.10.attention.attention.key.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.10.attention.attention.key.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.10.attention.attention.value.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.10.attention.attention.value.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.10.attention.output.dense.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.10.attention.output.dense.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.10.layernorm_after.weight", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.10.layernorm_after.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.10.intermediate.dense.weight", pybuda.Parameter(*(3072, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.10.intermediate.dense.bias", pybuda.Parameter(*(3072,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.10.output.dense.weight", pybuda.Parameter(*(768, 3072), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.10.output.dense.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.11.layernorm_before.weight", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.11.layernorm_before.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.11.attention.attention.query.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.11.attention.attention.query.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.11.attention.attention.key.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.11.attention.attention.key.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.11.attention.attention.value.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.11.attention.attention.value.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.11.attention.output.dense.weight", pybuda.Parameter(*(768, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.11.attention.output.dense.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.11.layernorm_after.weight", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.11.layernorm_after.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.11.intermediate.dense.weight", pybuda.Parameter(*(3072, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.11.intermediate.dense.bias", pybuda.Parameter(*(3072,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.11.output.dense.weight", pybuda.Parameter(*(768, 3072), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.encoder.layer.11.output.dense.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.layernorm.weight", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("vit.layernorm.bias", pybuda.Parameter(*(768,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("classifier.weight", pybuda.Parameter(*(26, 768), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_parameter("classifier.bias", pybuda.Parameter(*(26,), requires_grad=True, dev_data_format=pybuda.DataFormat.Float32))
        self.add_constant("const_00", shape=(1,))
        self.add_constant("const_10", shape=(1,))
        self.add_constant("const_20", shape=(1,))
        self.add_constant("const_30", shape=(1,))
        self.add_constant("const_40", shape=(1,))
        self.add_constant("const_50", shape=(1,))
        self.add_constant("const_60", shape=(1,))
        self.add_constant("const_70", shape=(1,))
        self.add_constant("const_80", shape=(1,))
        self.add_constant("const_90", shape=(1,))
        self.add_constant("const_100", shape=(1,))
        self.add_constant("const_110", shape=(1,))

    def forward(self, pixel_values):
        broadcast_201 = pybuda.op.Broadcast("", self.get_parameter("vit.embeddings.cls_token"), dim=-3, shape=1).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEmbeddings::embeddings")
        conv2d_202 = pybuda.op.Conv2d("", pixel_values, self.get_parameter("vit.embeddings.patch_embeddings.projection.weight"), self.get_parameter("vit.embeddings.patch_embeddings.projection.bias"), stride=[16, 16], padding=[0, 0, 0, 0], dilation=1, groups=1, channel_last=0)
        reshape_203 = pybuda.op.Reshape("", conv2d_202, shape=(1, 768, 196)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEmbeddings::embeddings/transformers.models.vit.modeling_vit.ViTPatchEmbeddings::patch_embeddings")
        conv2d_202._value = None
        transpose_204 = pybuda.op.Transpose("", reshape_203, dim0=-2, dim1=-1, out_dtype=torch.float32).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEmbeddings::embeddings/transformers.models.vit.modeling_vit.ViTPatchEmbeddings::patch_embeddings")
        reshape_203._value = None
        concatenate_205 = pybuda.op.Concatenate("", broadcast_201, transpose_204, axis=-2).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEmbeddings::embeddings")
        broadcast_201._value = None
        transpose_204._value = None
        add_206 = pybuda.op.Add("", concatenate_205, self.get_parameter("vit.embeddings.position_embeddings")).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEmbeddings::embeddings")
        concatenate_205._value = None
        dropout_207 = pybuda.op.Identity("", add_206).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEmbeddings::embeddings/torch.nn.modules.dropout.Dropout::dropout")
        add_206._value = None
        layernorm_208 = pybuda.op.Layernorm("", dropout_207, self.get_parameter("vit.encoder.layer.0.layernorm_before.weight"), self.get_parameter("vit.encoder.layer.0.layernorm_before.bias"), dim=-1, epsilon=0.0).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0/torch.nn.modules.normalization.LayerNorm::layernorm_before")
        reshape_209 = pybuda.op.Reshape("", layernorm_208, shape=(197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::query")
        layernorm_208._value = None
        transpose_210 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.0.attention.attention.query.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_211 = pybuda.op.Matmul("", reshape_209, transpose_210).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::query")
        transpose_210._value = None
        reshape_212 = pybuda.op.Reshape("", matmul_211, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::query")
        matmul_211._value = None
        add_213 = pybuda.op.Add("", reshape_212, self.get_parameter("vit.encoder.layer.0.attention.attention.query.bias"))
        reshape_212._value = None
        hslice_214 = pybuda.op.HSlice("", add_213, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        add_213._value = None
        reshape_215 = pybuda.op.Reshape("", hslice_214, shape=(12, 197, 64)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        hslice_214._value = None
        transpose_216 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.0.attention.attention.key.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_217 = pybuda.op.Matmul("", reshape_209, transpose_216).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::key")
        transpose_216._value = None
        reshape_218 = pybuda.op.Reshape("", matmul_217, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::key")
        matmul_217._value = None
        add_219 = pybuda.op.Add("", reshape_218, self.get_parameter("vit.encoder.layer.0.attention.attention.key.bias"))
        reshape_218._value = None
        hslice_220 = pybuda.op.HSlice("", add_219, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        add_219._value = None
        reshape_221 = pybuda.op.Reshape("", hslice_220, shape=(12, 197, 64)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        hslice_220._value = None
        transpose_222 = pybuda.op.Transpose("", reshape_221, dim0=-2, dim1=-1, out_dtype=torch.float32)
        reshape_221._value = None
        matmul_223 = pybuda.op.Matmul("", reshape_215, transpose_222).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        reshape_215._value = None
        transpose_222._value = None
        reshape_224 = pybuda.op.Reshape("", matmul_223, shape=(1, 12, 197, 197)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        matmul_223._value = None
        multiply_226 = pybuda.op.Multiply("", reshape_224, self.get_constant("const_00")).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        reshape_224._value = None
        softmax_227 = pybuda.op.Softmax("", multiply_226, dim=-1).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        multiply_226._value = None
        dropout_228 = pybuda.op.Identity("", softmax_227).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.dropout.Dropout::dropout")
        softmax_227._value = None
        reshape_229 = pybuda.op.Reshape("", dropout_228, shape=(12, 197, 197)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        dropout_228._value = None
        transpose_230 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.0.attention.attention.value.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_231 = pybuda.op.Matmul("", reshape_209, transpose_230).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::value")
        reshape_209._value = None
        transpose_230._value = None
        reshape_232 = pybuda.op.Reshape("", matmul_231, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::value")
        matmul_231._value = None
        add_233 = pybuda.op.Add("", reshape_232, self.get_parameter("vit.encoder.layer.0.attention.attention.value.bias"))
        reshape_232._value = None
        hslice_234 = pybuda.op.HSlice("", add_233, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        add_233._value = None
        transpose_235 = pybuda.op.Transpose("", hslice_234, dim0=-2, dim1=-1, out_dtype=torch.float32).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        hslice_234._value = None
        reshape_236 = pybuda.op.Reshape("", transpose_235, shape=(12, 64, 197)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        transpose_235._value = None
        transpose_237 = pybuda.op.Transpose("", reshape_236, dim0=-2, dim1=-1, out_dtype=torch.float32)
        reshape_236._value = None
        matmul_238 = pybuda.op.Matmul("", reshape_229, transpose_237).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        reshape_229._value = None
        transpose_237._value = None
        reshape_239 = pybuda.op.Reshape("", matmul_238, shape=(1, 12, 197, 64)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        matmul_238._value = None
        hstack_240 = pybuda.op.HStack("", reshape_239, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.linear.Linear::dense")
        reshape_239._value = None
        transpose_241 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.0.attention.output.dense.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_242 = pybuda.op.Matmul("", hstack_240, transpose_241).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.linear.Linear::dense")
        hstack_240._value = None
        transpose_241._value = None
        reshape_243 = pybuda.op.Reshape("", matmul_242, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.linear.Linear::dense")
        matmul_242._value = None
        add_244 = pybuda.op.Add("", reshape_243, self.get_parameter("vit.encoder.layer.0.attention.output.dense.bias"))
        reshape_243._value = None
        dropout_245 = pybuda.op.Identity("", add_244).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.dropout.Dropout::dropout")
        add_244._value = None
        add_246 = pybuda.op.Add("", dropout_245, dropout_207).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0")
        dropout_245._value = None
        dropout_207._value = None
        layernorm_247 = pybuda.op.Layernorm("", add_246, self.get_parameter("vit.encoder.layer.0.layernorm_after.weight"), self.get_parameter("vit.encoder.layer.0.layernorm_after.bias"), dim=-1, epsilon=0.0).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0/torch.nn.modules.normalization.LayerNorm::layernorm_after")
        reshape_248 = pybuda.op.Reshape("", layernorm_247, shape=(197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/torch.nn.modules.linear.Linear::dense")
        layernorm_247._value = None
        transpose_249 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.0.intermediate.dense.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_250 = pybuda.op.Matmul("", reshape_248, transpose_249).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/torch.nn.modules.linear.Linear::dense")
        reshape_248._value = None
        transpose_249._value = None
        reshape_251 = pybuda.op.Reshape("", matmul_250, shape=(1, 197, 3072)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/torch.nn.modules.linear.Linear::dense")
        matmul_250._value = None
        add_252 = pybuda.op.Add("", reshape_251, self.get_parameter("vit.encoder.layer.0.intermediate.dense.bias"))
        reshape_251._value = None
        gelu_253 = pybuda.op.Gelu("", add_252, approximate="none").set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/transformers.activations.GELUActivation::intermediate_act_fn")
        add_252._value = None
        reshape_254 = pybuda.op.Reshape("", gelu_253, shape=(197, 3072)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.linear.Linear::dense")
        gelu_253._value = None
        transpose_255 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.0.output.dense.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_256 = pybuda.op.Matmul("", reshape_254, transpose_255).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.linear.Linear::dense")
        reshape_254._value = None
        transpose_255._value = None
        reshape_257 = pybuda.op.Reshape("", matmul_256, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.linear.Linear::dense")
        matmul_256._value = None
        add_258 = pybuda.op.Add("", reshape_257, self.get_parameter("vit.encoder.layer.0.output.dense.bias"))
        reshape_257._value = None
        dropout_259 = pybuda.op.Identity("", add_258).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.dropout.Dropout::dropout")
        add_258._value = None
        add_260 = pybuda.op.Add("", dropout_259, add_246).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.0/transformers.models.vit.modeling_vit.ViTOutput::output")
        dropout_259._value = None
        add_246._value = None
        layernorm_261 = pybuda.op.Layernorm("", add_260, self.get_parameter("vit.encoder.layer.1.layernorm_before.weight"), self.get_parameter("vit.encoder.layer.1.layernorm_before.bias"), dim=-1, epsilon=0.0).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1/torch.nn.modules.normalization.LayerNorm::layernorm_before")
        reshape_262 = pybuda.op.Reshape("", layernorm_261, shape=(197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::query")
        layernorm_261._value = None
        transpose_263 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.1.attention.attention.query.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_264 = pybuda.op.Matmul("", reshape_262, transpose_263).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::query")
        transpose_263._value = None
        reshape_265 = pybuda.op.Reshape("", matmul_264, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::query")
        matmul_264._value = None
        add_266 = pybuda.op.Add("", reshape_265, self.get_parameter("vit.encoder.layer.1.attention.attention.query.bias"))
        reshape_265._value = None
        hslice_267 = pybuda.op.HSlice("", add_266, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        add_266._value = None
        reshape_268 = pybuda.op.Reshape("", hslice_267, shape=(12, 197, 64)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        hslice_267._value = None
        transpose_269 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.1.attention.attention.key.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_270 = pybuda.op.Matmul("", reshape_262, transpose_269).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::key")
        transpose_269._value = None
        reshape_271 = pybuda.op.Reshape("", matmul_270, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::key")
        matmul_270._value = None
        add_272 = pybuda.op.Add("", reshape_271, self.get_parameter("vit.encoder.layer.1.attention.attention.key.bias"))
        reshape_271._value = None
        hslice_273 = pybuda.op.HSlice("", add_272, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        add_272._value = None
        reshape_274 = pybuda.op.Reshape("", hslice_273, shape=(12, 197, 64)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        hslice_273._value = None
        transpose_275 = pybuda.op.Transpose("", reshape_274, dim0=-2, dim1=-1, out_dtype=torch.float32)
        reshape_274._value = None
        matmul_276 = pybuda.op.Matmul("", reshape_268, transpose_275).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        reshape_268._value = None
        transpose_275._value = None
        reshape_277 = pybuda.op.Reshape("", matmul_276, shape=(1, 12, 197, 197)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        matmul_276._value = None
        multiply_279 = pybuda.op.Multiply("", reshape_277, self.get_constant("const_10")).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        reshape_277._value = None
        softmax_280 = pybuda.op.Softmax("", multiply_279, dim=-1).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        multiply_279._value = None
        dropout_281 = pybuda.op.Identity("", softmax_280).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.dropout.Dropout::dropout")
        softmax_280._value = None
        reshape_282 = pybuda.op.Reshape("", dropout_281, shape=(12, 197, 197)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        dropout_281._value = None
        transpose_283 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.1.attention.attention.value.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_284 = pybuda.op.Matmul("", reshape_262, transpose_283).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::value")
        reshape_262._value = None
        transpose_283._value = None
        reshape_285 = pybuda.op.Reshape("", matmul_284, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::value")
        matmul_284._value = None
        add_286 = pybuda.op.Add("", reshape_285, self.get_parameter("vit.encoder.layer.1.attention.attention.value.bias"))
        reshape_285._value = None
        hslice_287 = pybuda.op.HSlice("", add_286, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        add_286._value = None
        transpose_288 = pybuda.op.Transpose("", hslice_287, dim0=-2, dim1=-1, out_dtype=torch.float32).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        hslice_287._value = None
        reshape_289 = pybuda.op.Reshape("", transpose_288, shape=(12, 64, 197)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        transpose_288._value = None
        transpose_290 = pybuda.op.Transpose("", reshape_289, dim0=-2, dim1=-1, out_dtype=torch.float32)
        reshape_289._value = None
        matmul_291 = pybuda.op.Matmul("", reshape_282, transpose_290).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        reshape_282._value = None
        transpose_290._value = None
        reshape_292 = pybuda.op.Reshape("", matmul_291, shape=(1, 12, 197, 64)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        matmul_291._value = None
        hstack_293 = pybuda.op.HStack("", reshape_292, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.linear.Linear::dense")
        reshape_292._value = None
        transpose_294 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.1.attention.output.dense.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_295 = pybuda.op.Matmul("", hstack_293, transpose_294).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.linear.Linear::dense")
        hstack_293._value = None
        transpose_294._value = None
        reshape_296 = pybuda.op.Reshape("", matmul_295, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.linear.Linear::dense")
        matmul_295._value = None
        add_297 = pybuda.op.Add("", reshape_296, self.get_parameter("vit.encoder.layer.1.attention.output.dense.bias"))
        reshape_296._value = None
        dropout_298 = pybuda.op.Identity("", add_297).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.dropout.Dropout::dropout")
        add_297._value = None
        add_299 = pybuda.op.Add("", dropout_298, add_260).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1")
        dropout_298._value = None
        add_260._value = None
        layernorm_300 = pybuda.op.Layernorm("", add_299, self.get_parameter("vit.encoder.layer.1.layernorm_after.weight"), self.get_parameter("vit.encoder.layer.1.layernorm_after.bias"), dim=-1, epsilon=0.0).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1/torch.nn.modules.normalization.LayerNorm::layernorm_after")
        reshape_301 = pybuda.op.Reshape("", layernorm_300, shape=(197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/torch.nn.modules.linear.Linear::dense")
        layernorm_300._value = None
        transpose_302 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.1.intermediate.dense.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_303 = pybuda.op.Matmul("", reshape_301, transpose_302).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/torch.nn.modules.linear.Linear::dense")
        reshape_301._value = None
        transpose_302._value = None
        reshape_304 = pybuda.op.Reshape("", matmul_303, shape=(1, 197, 3072)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/torch.nn.modules.linear.Linear::dense")
        matmul_303._value = None
        add_305 = pybuda.op.Add("", reshape_304, self.get_parameter("vit.encoder.layer.1.intermediate.dense.bias"))
        reshape_304._value = None
        gelu_306 = pybuda.op.Gelu("", add_305, approximate="none").set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/transformers.activations.GELUActivation::intermediate_act_fn")
        add_305._value = None
        reshape_307 = pybuda.op.Reshape("", gelu_306, shape=(197, 3072)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.linear.Linear::dense")
        gelu_306._value = None
        transpose_308 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.1.output.dense.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_309 = pybuda.op.Matmul("", reshape_307, transpose_308).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.linear.Linear::dense")
        reshape_307._value = None
        transpose_308._value = None
        reshape_310 = pybuda.op.Reshape("", matmul_309, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.linear.Linear::dense")
        matmul_309._value = None
        add_311 = pybuda.op.Add("", reshape_310, self.get_parameter("vit.encoder.layer.1.output.dense.bias"))
        reshape_310._value = None
        dropout_312 = pybuda.op.Identity("", add_311).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.dropout.Dropout::dropout")
        add_311._value = None
        add_313 = pybuda.op.Add("", dropout_312, add_299).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.1/transformers.models.vit.modeling_vit.ViTOutput::output")
        dropout_312._value = None
        add_299._value = None
        layernorm_314 = pybuda.op.Layernorm("", add_313, self.get_parameter("vit.encoder.layer.2.layernorm_before.weight"), self.get_parameter("vit.encoder.layer.2.layernorm_before.bias"), dim=-1, epsilon=0.0).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2/torch.nn.modules.normalization.LayerNorm::layernorm_before")
        reshape_315 = pybuda.op.Reshape("", layernorm_314, shape=(197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::query")
        layernorm_314._value = None
        transpose_316 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.2.attention.attention.query.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_317 = pybuda.op.Matmul("", reshape_315, transpose_316).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::query")
        transpose_316._value = None
        reshape_318 = pybuda.op.Reshape("", matmul_317, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::query")
        matmul_317._value = None
        add_319 = pybuda.op.Add("", reshape_318, self.get_parameter("vit.encoder.layer.2.attention.attention.query.bias"))
        reshape_318._value = None
        hslice_320 = pybuda.op.HSlice("", add_319, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        add_319._value = None
        reshape_321 = pybuda.op.Reshape("", hslice_320, shape=(12, 197, 64)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        hslice_320._value = None
        transpose_322 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.2.attention.attention.key.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_323 = pybuda.op.Matmul("", reshape_315, transpose_322).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::key")
        transpose_322._value = None
        reshape_324 = pybuda.op.Reshape("", matmul_323, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::key")
        matmul_323._value = None
        add_325 = pybuda.op.Add("", reshape_324, self.get_parameter("vit.encoder.layer.2.attention.attention.key.bias"))
        reshape_324._value = None
        hslice_326 = pybuda.op.HSlice("", add_325, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        add_325._value = None
        reshape_327 = pybuda.op.Reshape("", hslice_326, shape=(12, 197, 64)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        hslice_326._value = None
        transpose_328 = pybuda.op.Transpose("", reshape_327, dim0=-2, dim1=-1, out_dtype=torch.float32)
        reshape_327._value = None
        matmul_329 = pybuda.op.Matmul("", reshape_321, transpose_328).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        reshape_321._value = None
        transpose_328._value = None
        reshape_330 = pybuda.op.Reshape("", matmul_329, shape=(1, 12, 197, 197)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        matmul_329._value = None
        multiply_332 = pybuda.op.Multiply("", reshape_330, self.get_constant("const_20")).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        reshape_330._value = None
        softmax_333 = pybuda.op.Softmax("", multiply_332, dim=-1).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        multiply_332._value = None
        dropout_334 = pybuda.op.Identity("", softmax_333).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.dropout.Dropout::dropout")
        softmax_333._value = None
        reshape_335 = pybuda.op.Reshape("", dropout_334, shape=(12, 197, 197)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        dropout_334._value = None
        transpose_336 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.2.attention.attention.value.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_337 = pybuda.op.Matmul("", reshape_315, transpose_336).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::value")
        reshape_315._value = None
        transpose_336._value = None
        reshape_338 = pybuda.op.Reshape("", matmul_337, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::value")
        matmul_337._value = None
        add_339 = pybuda.op.Add("", reshape_338, self.get_parameter("vit.encoder.layer.2.attention.attention.value.bias"))
        reshape_338._value = None
        hslice_340 = pybuda.op.HSlice("", add_339, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        add_339._value = None
        transpose_341 = pybuda.op.Transpose("", hslice_340, dim0=-2, dim1=-1, out_dtype=torch.float32).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        hslice_340._value = None
        reshape_342 = pybuda.op.Reshape("", transpose_341, shape=(12, 64, 197)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        transpose_341._value = None
        transpose_343 = pybuda.op.Transpose("", reshape_342, dim0=-2, dim1=-1, out_dtype=torch.float32)
        reshape_342._value = None
        matmul_344 = pybuda.op.Matmul("", reshape_335, transpose_343).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        reshape_335._value = None
        transpose_343._value = None
        reshape_345 = pybuda.op.Reshape("", matmul_344, shape=(1, 12, 197, 64)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        matmul_344._value = None
        hstack_346 = pybuda.op.HStack("", reshape_345, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.linear.Linear::dense")
        reshape_345._value = None
        transpose_347 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.2.attention.output.dense.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_348 = pybuda.op.Matmul("", hstack_346, transpose_347).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.linear.Linear::dense")
        hstack_346._value = None
        transpose_347._value = None
        reshape_349 = pybuda.op.Reshape("", matmul_348, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.linear.Linear::dense")
        matmul_348._value = None
        add_350 = pybuda.op.Add("", reshape_349, self.get_parameter("vit.encoder.layer.2.attention.output.dense.bias"))
        reshape_349._value = None
        dropout_351 = pybuda.op.Identity("", add_350).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.dropout.Dropout::dropout")
        add_350._value = None
        add_352 = pybuda.op.Add("", dropout_351, add_313).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2")
        dropout_351._value = None
        add_313._value = None
        layernorm_353 = pybuda.op.Layernorm("", add_352, self.get_parameter("vit.encoder.layer.2.layernorm_after.weight"), self.get_parameter("vit.encoder.layer.2.layernorm_after.bias"), dim=-1, epsilon=0.0).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2/torch.nn.modules.normalization.LayerNorm::layernorm_after")
        reshape_354 = pybuda.op.Reshape("", layernorm_353, shape=(197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/torch.nn.modules.linear.Linear::dense")
        layernorm_353._value = None
        transpose_355 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.2.intermediate.dense.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_356 = pybuda.op.Matmul("", reshape_354, transpose_355).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/torch.nn.modules.linear.Linear::dense")
        reshape_354._value = None
        transpose_355._value = None
        reshape_357 = pybuda.op.Reshape("", matmul_356, shape=(1, 197, 3072)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/torch.nn.modules.linear.Linear::dense")
        matmul_356._value = None
        add_358 = pybuda.op.Add("", reshape_357, self.get_parameter("vit.encoder.layer.2.intermediate.dense.bias"))
        reshape_357._value = None
        gelu_359 = pybuda.op.Gelu("", add_358, approximate="none").set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/transformers.activations.GELUActivation::intermediate_act_fn")
        add_358._value = None
        reshape_360 = pybuda.op.Reshape("", gelu_359, shape=(197, 3072)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.linear.Linear::dense")
        gelu_359._value = None
        transpose_361 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.2.output.dense.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_362 = pybuda.op.Matmul("", reshape_360, transpose_361).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.linear.Linear::dense")
        reshape_360._value = None
        transpose_361._value = None
        reshape_363 = pybuda.op.Reshape("", matmul_362, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.linear.Linear::dense")
        matmul_362._value = None
        add_364 = pybuda.op.Add("", reshape_363, self.get_parameter("vit.encoder.layer.2.output.dense.bias"))
        reshape_363._value = None
        dropout_365 = pybuda.op.Identity("", add_364).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.dropout.Dropout::dropout")
        add_364._value = None
        add_366 = pybuda.op.Add("", dropout_365, add_352).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.2/transformers.models.vit.modeling_vit.ViTOutput::output")
        dropout_365._value = None
        add_352._value = None
        layernorm_367 = pybuda.op.Layernorm("", add_366, self.get_parameter("vit.encoder.layer.3.layernorm_before.weight"), self.get_parameter("vit.encoder.layer.3.layernorm_before.bias"), dim=-1, epsilon=0.0).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3/torch.nn.modules.normalization.LayerNorm::layernorm_before")
        reshape_368 = pybuda.op.Reshape("", layernorm_367, shape=(197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::query")
        layernorm_367._value = None
        transpose_369 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.3.attention.attention.query.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_370 = pybuda.op.Matmul("", reshape_368, transpose_369).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::query")
        transpose_369._value = None
        reshape_371 = pybuda.op.Reshape("", matmul_370, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::query")
        matmul_370._value = None
        add_372 = pybuda.op.Add("", reshape_371, self.get_parameter("vit.encoder.layer.3.attention.attention.query.bias"))
        reshape_371._value = None
        hslice_373 = pybuda.op.HSlice("", add_372, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        add_372._value = None
        reshape_374 = pybuda.op.Reshape("", hslice_373, shape=(12, 197, 64)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        hslice_373._value = None
        transpose_375 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.3.attention.attention.key.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_376 = pybuda.op.Matmul("", reshape_368, transpose_375).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::key")
        transpose_375._value = None
        reshape_377 = pybuda.op.Reshape("", matmul_376, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::key")
        matmul_376._value = None
        add_378 = pybuda.op.Add("", reshape_377, self.get_parameter("vit.encoder.layer.3.attention.attention.key.bias"))
        reshape_377._value = None
        hslice_379 = pybuda.op.HSlice("", add_378, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        add_378._value = None
        reshape_380 = pybuda.op.Reshape("", hslice_379, shape=(12, 197, 64)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        hslice_379._value = None
        transpose_381 = pybuda.op.Transpose("", reshape_380, dim0=-2, dim1=-1, out_dtype=torch.float32)
        reshape_380._value = None
        matmul_382 = pybuda.op.Matmul("", reshape_374, transpose_381).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        reshape_374._value = None
        transpose_381._value = None
        reshape_383 = pybuda.op.Reshape("", matmul_382, shape=(1, 12, 197, 197)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        matmul_382._value = None
        multiply_385 = pybuda.op.Multiply("", reshape_383, self.get_constant("const_30")).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        reshape_383._value = None
        softmax_386 = pybuda.op.Softmax("", multiply_385, dim=-1).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        multiply_385._value = None
        dropout_387 = pybuda.op.Identity("", softmax_386).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.dropout.Dropout::dropout")
        softmax_386._value = None
        reshape_388 = pybuda.op.Reshape("", dropout_387, shape=(12, 197, 197)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        dropout_387._value = None
        transpose_389 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.3.attention.attention.value.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_390 = pybuda.op.Matmul("", reshape_368, transpose_389).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::value")
        reshape_368._value = None
        transpose_389._value = None
        reshape_391 = pybuda.op.Reshape("", matmul_390, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::value")
        matmul_390._value = None
        add_392 = pybuda.op.Add("", reshape_391, self.get_parameter("vit.encoder.layer.3.attention.attention.value.bias"))
        reshape_391._value = None
        hslice_393 = pybuda.op.HSlice("", add_392, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        add_392._value = None
        transpose_394 = pybuda.op.Transpose("", hslice_393, dim0=-2, dim1=-1, out_dtype=torch.float32).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        hslice_393._value = None
        reshape_395 = pybuda.op.Reshape("", transpose_394, shape=(12, 64, 197)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        transpose_394._value = None
        transpose_396 = pybuda.op.Transpose("", reshape_395, dim0=-2, dim1=-1, out_dtype=torch.float32)
        reshape_395._value = None
        matmul_397 = pybuda.op.Matmul("", reshape_388, transpose_396).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        reshape_388._value = None
        transpose_396._value = None
        reshape_398 = pybuda.op.Reshape("", matmul_397, shape=(1, 12, 197, 64)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        matmul_397._value = None
        hstack_399 = pybuda.op.HStack("", reshape_398, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.linear.Linear::dense")
        reshape_398._value = None
        transpose_400 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.3.attention.output.dense.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_401 = pybuda.op.Matmul("", hstack_399, transpose_400).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.linear.Linear::dense")
        hstack_399._value = None
        transpose_400._value = None
        reshape_402 = pybuda.op.Reshape("", matmul_401, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.linear.Linear::dense")
        matmul_401._value = None
        add_403 = pybuda.op.Add("", reshape_402, self.get_parameter("vit.encoder.layer.3.attention.output.dense.bias"))
        reshape_402._value = None
        dropout_404 = pybuda.op.Identity("", add_403).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.dropout.Dropout::dropout")
        add_403._value = None
        add_405 = pybuda.op.Add("", dropout_404, add_366).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3")
        dropout_404._value = None
        add_366._value = None
        layernorm_406 = pybuda.op.Layernorm("", add_405, self.get_parameter("vit.encoder.layer.3.layernorm_after.weight"), self.get_parameter("vit.encoder.layer.3.layernorm_after.bias"), dim=-1, epsilon=0.0).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3/torch.nn.modules.normalization.LayerNorm::layernorm_after")
        reshape_407 = pybuda.op.Reshape("", layernorm_406, shape=(197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/torch.nn.modules.linear.Linear::dense")
        layernorm_406._value = None
        transpose_408 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.3.intermediate.dense.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_409 = pybuda.op.Matmul("", reshape_407, transpose_408).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/torch.nn.modules.linear.Linear::dense")
        reshape_407._value = None
        transpose_408._value = None
        reshape_410 = pybuda.op.Reshape("", matmul_409, shape=(1, 197, 3072)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/torch.nn.modules.linear.Linear::dense")
        matmul_409._value = None
        add_411 = pybuda.op.Add("", reshape_410, self.get_parameter("vit.encoder.layer.3.intermediate.dense.bias"))
        reshape_410._value = None
        gelu_412 = pybuda.op.Gelu("", add_411, approximate="none").set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/transformers.activations.GELUActivation::intermediate_act_fn")
        add_411._value = None
        reshape_413 = pybuda.op.Reshape("", gelu_412, shape=(197, 3072)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.linear.Linear::dense")
        gelu_412._value = None
        transpose_414 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.3.output.dense.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_415 = pybuda.op.Matmul("", reshape_413, transpose_414).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.linear.Linear::dense")
        reshape_413._value = None
        transpose_414._value = None
        reshape_416 = pybuda.op.Reshape("", matmul_415, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.linear.Linear::dense")
        matmul_415._value = None
        add_417 = pybuda.op.Add("", reshape_416, self.get_parameter("vit.encoder.layer.3.output.dense.bias"))
        reshape_416._value = None
        dropout_418 = pybuda.op.Identity("", add_417).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.dropout.Dropout::dropout")
        add_417._value = None
        add_419 = pybuda.op.Add("", dropout_418, add_405).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.3/transformers.models.vit.modeling_vit.ViTOutput::output")
        dropout_418._value = None
        add_405._value = None
        layernorm_420 = pybuda.op.Layernorm("", add_419, self.get_parameter("vit.encoder.layer.4.layernorm_before.weight"), self.get_parameter("vit.encoder.layer.4.layernorm_before.bias"), dim=-1, epsilon=0.0).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4/torch.nn.modules.normalization.LayerNorm::layernorm_before")
        reshape_421 = pybuda.op.Reshape("", layernorm_420, shape=(197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::query")
        layernorm_420._value = None
        transpose_422 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.4.attention.attention.query.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_423 = pybuda.op.Matmul("", reshape_421, transpose_422).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::query")
        transpose_422._value = None
        reshape_424 = pybuda.op.Reshape("", matmul_423, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::query")
        matmul_423._value = None
        add_425 = pybuda.op.Add("", reshape_424, self.get_parameter("vit.encoder.layer.4.attention.attention.query.bias"))
        reshape_424._value = None
        hslice_426 = pybuda.op.HSlice("", add_425, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        add_425._value = None
        reshape_427 = pybuda.op.Reshape("", hslice_426, shape=(12, 197, 64)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        hslice_426._value = None
        transpose_428 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.4.attention.attention.key.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_429 = pybuda.op.Matmul("", reshape_421, transpose_428).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::key")
        transpose_428._value = None
        reshape_430 = pybuda.op.Reshape("", matmul_429, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::key")
        matmul_429._value = None
        add_431 = pybuda.op.Add("", reshape_430, self.get_parameter("vit.encoder.layer.4.attention.attention.key.bias"))
        reshape_430._value = None
        hslice_432 = pybuda.op.HSlice("", add_431, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        add_431._value = None
        reshape_433 = pybuda.op.Reshape("", hslice_432, shape=(12, 197, 64)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        hslice_432._value = None
        transpose_434 = pybuda.op.Transpose("", reshape_433, dim0=-2, dim1=-1, out_dtype=torch.float32)
        reshape_433._value = None
        matmul_435 = pybuda.op.Matmul("", reshape_427, transpose_434).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        reshape_427._value = None
        transpose_434._value = None
        reshape_436 = pybuda.op.Reshape("", matmul_435, shape=(1, 12, 197, 197)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        matmul_435._value = None
        multiply_438 = pybuda.op.Multiply("", reshape_436, self.get_constant("const_40")).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        reshape_436._value = None
        softmax_439 = pybuda.op.Softmax("", multiply_438, dim=-1).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        multiply_438._value = None
        dropout_440 = pybuda.op.Identity("", softmax_439).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.dropout.Dropout::dropout")
        softmax_439._value = None
        reshape_441 = pybuda.op.Reshape("", dropout_440, shape=(12, 197, 197)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        dropout_440._value = None
        transpose_442 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.4.attention.attention.value.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_443 = pybuda.op.Matmul("", reshape_421, transpose_442).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::value")
        reshape_421._value = None
        transpose_442._value = None
        reshape_444 = pybuda.op.Reshape("", matmul_443, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::value")
        matmul_443._value = None
        add_445 = pybuda.op.Add("", reshape_444, self.get_parameter("vit.encoder.layer.4.attention.attention.value.bias"))
        reshape_444._value = None
        hslice_446 = pybuda.op.HSlice("", add_445, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        add_445._value = None
        transpose_447 = pybuda.op.Transpose("", hslice_446, dim0=-2, dim1=-1, out_dtype=torch.float32).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        hslice_446._value = None
        reshape_448 = pybuda.op.Reshape("", transpose_447, shape=(12, 64, 197)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        transpose_447._value = None
        transpose_449 = pybuda.op.Transpose("", reshape_448, dim0=-2, dim1=-1, out_dtype=torch.float32)
        reshape_448._value = None
        matmul_450 = pybuda.op.Matmul("", reshape_441, transpose_449).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        reshape_441._value = None
        transpose_449._value = None
        reshape_451 = pybuda.op.Reshape("", matmul_450, shape=(1, 12, 197, 64)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        matmul_450._value = None
        hstack_452 = pybuda.op.HStack("", reshape_451, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.linear.Linear::dense")
        reshape_451._value = None
        transpose_453 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.4.attention.output.dense.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_454 = pybuda.op.Matmul("", hstack_452, transpose_453).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.linear.Linear::dense")
        hstack_452._value = None
        transpose_453._value = None
        reshape_455 = pybuda.op.Reshape("", matmul_454, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.linear.Linear::dense")
        matmul_454._value = None
        add_456 = pybuda.op.Add("", reshape_455, self.get_parameter("vit.encoder.layer.4.attention.output.dense.bias"))
        reshape_455._value = None
        dropout_457 = pybuda.op.Identity("", add_456).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.dropout.Dropout::dropout")
        add_456._value = None
        add_458 = pybuda.op.Add("", dropout_457, add_419).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4")
        dropout_457._value = None
        add_419._value = None
        layernorm_459 = pybuda.op.Layernorm("", add_458, self.get_parameter("vit.encoder.layer.4.layernorm_after.weight"), self.get_parameter("vit.encoder.layer.4.layernorm_after.bias"), dim=-1, epsilon=0.0).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4/torch.nn.modules.normalization.LayerNorm::layernorm_after")
        reshape_460 = pybuda.op.Reshape("", layernorm_459, shape=(197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/torch.nn.modules.linear.Linear::dense")
        layernorm_459._value = None
        transpose_461 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.4.intermediate.dense.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_462 = pybuda.op.Matmul("", reshape_460, transpose_461).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/torch.nn.modules.linear.Linear::dense")
        reshape_460._value = None
        transpose_461._value = None
        reshape_463 = pybuda.op.Reshape("", matmul_462, shape=(1, 197, 3072)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/torch.nn.modules.linear.Linear::dense")
        matmul_462._value = None
        add_464 = pybuda.op.Add("", reshape_463, self.get_parameter("vit.encoder.layer.4.intermediate.dense.bias"))
        reshape_463._value = None
        gelu_465 = pybuda.op.Gelu("", add_464, approximate="none").set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/transformers.activations.GELUActivation::intermediate_act_fn")
        add_464._value = None
        reshape_466 = pybuda.op.Reshape("", gelu_465, shape=(197, 3072)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.linear.Linear::dense")
        gelu_465._value = None
        transpose_467 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.4.output.dense.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_468 = pybuda.op.Matmul("", reshape_466, transpose_467).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.linear.Linear::dense")
        reshape_466._value = None
        transpose_467._value = None
        reshape_469 = pybuda.op.Reshape("", matmul_468, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.linear.Linear::dense")
        matmul_468._value = None
        add_470 = pybuda.op.Add("", reshape_469, self.get_parameter("vit.encoder.layer.4.output.dense.bias"))
        reshape_469._value = None
        dropout_471 = pybuda.op.Identity("", add_470).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.dropout.Dropout::dropout")
        add_470._value = None
        add_472 = pybuda.op.Add("", dropout_471, add_458).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.4/transformers.models.vit.modeling_vit.ViTOutput::output")
        dropout_471._value = None
        add_458._value = None
        layernorm_473 = pybuda.op.Layernorm("", add_472, self.get_parameter("vit.encoder.layer.5.layernorm_before.weight"), self.get_parameter("vit.encoder.layer.5.layernorm_before.bias"), dim=-1, epsilon=0.0).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5/torch.nn.modules.normalization.LayerNorm::layernorm_before")
        reshape_474 = pybuda.op.Reshape("", layernorm_473, shape=(197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::query")
        layernorm_473._value = None
        transpose_475 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.5.attention.attention.query.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_476 = pybuda.op.Matmul("", reshape_474, transpose_475).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::query")
        transpose_475._value = None
        reshape_477 = pybuda.op.Reshape("", matmul_476, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::query")
        matmul_476._value = None
        add_478 = pybuda.op.Add("", reshape_477, self.get_parameter("vit.encoder.layer.5.attention.attention.query.bias"))
        reshape_477._value = None
        hslice_479 = pybuda.op.HSlice("", add_478, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        add_478._value = None
        reshape_480 = pybuda.op.Reshape("", hslice_479, shape=(12, 197, 64)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        hslice_479._value = None
        transpose_481 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.5.attention.attention.key.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_482 = pybuda.op.Matmul("", reshape_474, transpose_481).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::key")
        transpose_481._value = None
        reshape_483 = pybuda.op.Reshape("", matmul_482, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::key")
        matmul_482._value = None
        add_484 = pybuda.op.Add("", reshape_483, self.get_parameter("vit.encoder.layer.5.attention.attention.key.bias"))
        reshape_483._value = None
        hslice_485 = pybuda.op.HSlice("", add_484, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        add_484._value = None
        reshape_486 = pybuda.op.Reshape("", hslice_485, shape=(12, 197, 64)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        hslice_485._value = None
        transpose_487 = pybuda.op.Transpose("", reshape_486, dim0=-2, dim1=-1, out_dtype=torch.float32)
        reshape_486._value = None
        matmul_488 = pybuda.op.Matmul("", reshape_480, transpose_487).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        reshape_480._value = None
        transpose_487._value = None
        reshape_489 = pybuda.op.Reshape("", matmul_488, shape=(1, 12, 197, 197)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        matmul_488._value = None
        multiply_491 = pybuda.op.Multiply("", reshape_489, self.get_constant("const_50")).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        reshape_489._value = None
        softmax_492 = pybuda.op.Softmax("", multiply_491, dim=-1).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        multiply_491._value = None
        dropout_493 = pybuda.op.Identity("", softmax_492).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.dropout.Dropout::dropout")
        softmax_492._value = None
        reshape_494 = pybuda.op.Reshape("", dropout_493, shape=(12, 197, 197)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        dropout_493._value = None
        transpose_495 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.5.attention.attention.value.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_496 = pybuda.op.Matmul("", reshape_474, transpose_495).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::value")
        reshape_474._value = None
        transpose_495._value = None
        reshape_497 = pybuda.op.Reshape("", matmul_496, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::value")
        matmul_496._value = None
        add_498 = pybuda.op.Add("", reshape_497, self.get_parameter("vit.encoder.layer.5.attention.attention.value.bias"))
        reshape_497._value = None
        hslice_499 = pybuda.op.HSlice("", add_498, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        add_498._value = None
        transpose_500 = pybuda.op.Transpose("", hslice_499, dim0=-2, dim1=-1, out_dtype=torch.float32).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        hslice_499._value = None
        reshape_501 = pybuda.op.Reshape("", transpose_500, shape=(12, 64, 197)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        transpose_500._value = None
        transpose_502 = pybuda.op.Transpose("", reshape_501, dim0=-2, dim1=-1, out_dtype=torch.float32)
        reshape_501._value = None
        matmul_503 = pybuda.op.Matmul("", reshape_494, transpose_502).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        reshape_494._value = None
        transpose_502._value = None
        reshape_504 = pybuda.op.Reshape("", matmul_503, shape=(1, 12, 197, 64)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        matmul_503._value = None
        hstack_505 = pybuda.op.HStack("", reshape_504, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.linear.Linear::dense")
        reshape_504._value = None
        transpose_506 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.5.attention.output.dense.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_507 = pybuda.op.Matmul("", hstack_505, transpose_506).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.linear.Linear::dense")
        hstack_505._value = None
        transpose_506._value = None
        reshape_508 = pybuda.op.Reshape("", matmul_507, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.linear.Linear::dense")
        matmul_507._value = None
        add_509 = pybuda.op.Add("", reshape_508, self.get_parameter("vit.encoder.layer.5.attention.output.dense.bias"))
        reshape_508._value = None
        dropout_510 = pybuda.op.Identity("", add_509).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.dropout.Dropout::dropout")
        add_509._value = None
        add_511 = pybuda.op.Add("", dropout_510, add_472).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5")
        dropout_510._value = None
        add_472._value = None
        layernorm_512 = pybuda.op.Layernorm("", add_511, self.get_parameter("vit.encoder.layer.5.layernorm_after.weight"), self.get_parameter("vit.encoder.layer.5.layernorm_after.bias"), dim=-1, epsilon=0.0).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5/torch.nn.modules.normalization.LayerNorm::layernorm_after")
        reshape_513 = pybuda.op.Reshape("", layernorm_512, shape=(197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/torch.nn.modules.linear.Linear::dense")
        layernorm_512._value = None
        transpose_514 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.5.intermediate.dense.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_515 = pybuda.op.Matmul("", reshape_513, transpose_514).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/torch.nn.modules.linear.Linear::dense")
        reshape_513._value = None
        transpose_514._value = None
        reshape_516 = pybuda.op.Reshape("", matmul_515, shape=(1, 197, 3072)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/torch.nn.modules.linear.Linear::dense")
        matmul_515._value = None
        add_517 = pybuda.op.Add("", reshape_516, self.get_parameter("vit.encoder.layer.5.intermediate.dense.bias"))
        reshape_516._value = None
        gelu_518 = pybuda.op.Gelu("", add_517, approximate="none").set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/transformers.activations.GELUActivation::intermediate_act_fn")
        add_517._value = None
        reshape_519 = pybuda.op.Reshape("", gelu_518, shape=(197, 3072)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.linear.Linear::dense")
        gelu_518._value = None
        transpose_520 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.5.output.dense.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_521 = pybuda.op.Matmul("", reshape_519, transpose_520).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.linear.Linear::dense")
        reshape_519._value = None
        transpose_520._value = None
        reshape_522 = pybuda.op.Reshape("", matmul_521, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.linear.Linear::dense")
        matmul_521._value = None
        add_523 = pybuda.op.Add("", reshape_522, self.get_parameter("vit.encoder.layer.5.output.dense.bias"))
        reshape_522._value = None
        dropout_524 = pybuda.op.Identity("", add_523).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.dropout.Dropout::dropout")
        add_523._value = None
        add_525 = pybuda.op.Add("", dropout_524, add_511).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.5/transformers.models.vit.modeling_vit.ViTOutput::output")
        dropout_524._value = None
        add_511._value = None
        layernorm_526 = pybuda.op.Layernorm("", add_525, self.get_parameter("vit.encoder.layer.6.layernorm_before.weight"), self.get_parameter("vit.encoder.layer.6.layernorm_before.bias"), dim=-1, epsilon=0.0).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6/torch.nn.modules.normalization.LayerNorm::layernorm_before")
        reshape_527 = pybuda.op.Reshape("", layernorm_526, shape=(197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::query")
        layernorm_526._value = None
        transpose_528 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.6.attention.attention.query.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_529 = pybuda.op.Matmul("", reshape_527, transpose_528).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::query")
        transpose_528._value = None
        reshape_530 = pybuda.op.Reshape("", matmul_529, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::query")
        matmul_529._value = None
        add_531 = pybuda.op.Add("", reshape_530, self.get_parameter("vit.encoder.layer.6.attention.attention.query.bias"))
        reshape_530._value = None
        hslice_532 = pybuda.op.HSlice("", add_531, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        add_531._value = None
        reshape_533 = pybuda.op.Reshape("", hslice_532, shape=(12, 197, 64)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        hslice_532._value = None
        transpose_534 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.6.attention.attention.key.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_535 = pybuda.op.Matmul("", reshape_527, transpose_534).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::key")
        transpose_534._value = None
        reshape_536 = pybuda.op.Reshape("", matmul_535, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::key")
        matmul_535._value = None
        add_537 = pybuda.op.Add("", reshape_536, self.get_parameter("vit.encoder.layer.6.attention.attention.key.bias"))
        reshape_536._value = None
        hslice_538 = pybuda.op.HSlice("", add_537, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        add_537._value = None
        reshape_539 = pybuda.op.Reshape("", hslice_538, shape=(12, 197, 64)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        hslice_538._value = None
        transpose_540 = pybuda.op.Transpose("", reshape_539, dim0=-2, dim1=-1, out_dtype=torch.float32)
        reshape_539._value = None
        matmul_541 = pybuda.op.Matmul("", reshape_533, transpose_540).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        reshape_533._value = None
        transpose_540._value = None
        reshape_542 = pybuda.op.Reshape("", matmul_541, shape=(1, 12, 197, 197)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        matmul_541._value = None
        multiply_544 = pybuda.op.Multiply("", reshape_542, self.get_constant("const_60")).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        reshape_542._value = None
        softmax_545 = pybuda.op.Softmax("", multiply_544, dim=-1).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        multiply_544._value = None
        dropout_546 = pybuda.op.Identity("", softmax_545).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.dropout.Dropout::dropout")
        softmax_545._value = None
        reshape_547 = pybuda.op.Reshape("", dropout_546, shape=(12, 197, 197)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        dropout_546._value = None
        transpose_548 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.6.attention.attention.value.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_549 = pybuda.op.Matmul("", reshape_527, transpose_548).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::value")
        reshape_527._value = None
        transpose_548._value = None
        reshape_550 = pybuda.op.Reshape("", matmul_549, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::value")
        matmul_549._value = None
        add_551 = pybuda.op.Add("", reshape_550, self.get_parameter("vit.encoder.layer.6.attention.attention.value.bias"))
        reshape_550._value = None
        hslice_552 = pybuda.op.HSlice("", add_551, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        add_551._value = None
        transpose_553 = pybuda.op.Transpose("", hslice_552, dim0=-2, dim1=-1, out_dtype=torch.float32).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        hslice_552._value = None
        reshape_554 = pybuda.op.Reshape("", transpose_553, shape=(12, 64, 197)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        transpose_553._value = None
        transpose_555 = pybuda.op.Transpose("", reshape_554, dim0=-2, dim1=-1, out_dtype=torch.float32)
        reshape_554._value = None
        matmul_556 = pybuda.op.Matmul("", reshape_547, transpose_555).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        reshape_547._value = None
        transpose_555._value = None
        reshape_557 = pybuda.op.Reshape("", matmul_556, shape=(1, 12, 197, 64)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        matmul_556._value = None
        hstack_558 = pybuda.op.HStack("", reshape_557, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.linear.Linear::dense")
        reshape_557._value = None
        transpose_559 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.6.attention.output.dense.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_560 = pybuda.op.Matmul("", hstack_558, transpose_559).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.linear.Linear::dense")
        hstack_558._value = None
        transpose_559._value = None
        reshape_561 = pybuda.op.Reshape("", matmul_560, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.linear.Linear::dense")
        matmul_560._value = None
        add_562 = pybuda.op.Add("", reshape_561, self.get_parameter("vit.encoder.layer.6.attention.output.dense.bias"))
        reshape_561._value = None
        dropout_563 = pybuda.op.Identity("", add_562).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.dropout.Dropout::dropout")
        add_562._value = None
        add_564 = pybuda.op.Add("", dropout_563, add_525).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6")
        dropout_563._value = None
        add_525._value = None
        layernorm_565 = pybuda.op.Layernorm("", add_564, self.get_parameter("vit.encoder.layer.6.layernorm_after.weight"), self.get_parameter("vit.encoder.layer.6.layernorm_after.bias"), dim=-1, epsilon=0.0).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6/torch.nn.modules.normalization.LayerNorm::layernorm_after")
        reshape_566 = pybuda.op.Reshape("", layernorm_565, shape=(197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/torch.nn.modules.linear.Linear::dense")
        layernorm_565._value = None
        transpose_567 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.6.intermediate.dense.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_568 = pybuda.op.Matmul("", reshape_566, transpose_567).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/torch.nn.modules.linear.Linear::dense")
        reshape_566._value = None
        transpose_567._value = None
        reshape_569 = pybuda.op.Reshape("", matmul_568, shape=(1, 197, 3072)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/torch.nn.modules.linear.Linear::dense")
        matmul_568._value = None
        add_570 = pybuda.op.Add("", reshape_569, self.get_parameter("vit.encoder.layer.6.intermediate.dense.bias"))
        reshape_569._value = None
        gelu_571 = pybuda.op.Gelu("", add_570, approximate="none").set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/transformers.activations.GELUActivation::intermediate_act_fn")
        add_570._value = None
        reshape_572 = pybuda.op.Reshape("", gelu_571, shape=(197, 3072)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.linear.Linear::dense")
        gelu_571._value = None
        transpose_573 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.6.output.dense.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_574 = pybuda.op.Matmul("", reshape_572, transpose_573).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.linear.Linear::dense")
        reshape_572._value = None
        transpose_573._value = None
        reshape_575 = pybuda.op.Reshape("", matmul_574, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.linear.Linear::dense")
        matmul_574._value = None
        add_576 = pybuda.op.Add("", reshape_575, self.get_parameter("vit.encoder.layer.6.output.dense.bias"))
        reshape_575._value = None
        dropout_577 = pybuda.op.Identity("", add_576).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.dropout.Dropout::dropout")
        add_576._value = None
        add_578 = pybuda.op.Add("", dropout_577, add_564).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.6/transformers.models.vit.modeling_vit.ViTOutput::output")
        dropout_577._value = None
        add_564._value = None
        layernorm_579 = pybuda.op.Layernorm("", add_578, self.get_parameter("vit.encoder.layer.7.layernorm_before.weight"), self.get_parameter("vit.encoder.layer.7.layernorm_before.bias"), dim=-1, epsilon=0.0).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7/torch.nn.modules.normalization.LayerNorm::layernorm_before")
        reshape_580 = pybuda.op.Reshape("", layernorm_579, shape=(197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::query")
        layernorm_579._value = None
        transpose_581 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.7.attention.attention.query.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_582 = pybuda.op.Matmul("", reshape_580, transpose_581).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::query")
        transpose_581._value = None
        reshape_583 = pybuda.op.Reshape("", matmul_582, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::query")
        matmul_582._value = None
        add_584 = pybuda.op.Add("", reshape_583, self.get_parameter("vit.encoder.layer.7.attention.attention.query.bias"))
        reshape_583._value = None
        hslice_585 = pybuda.op.HSlice("", add_584, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        add_584._value = None
        reshape_586 = pybuda.op.Reshape("", hslice_585, shape=(12, 197, 64)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        hslice_585._value = None
        transpose_587 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.7.attention.attention.key.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_588 = pybuda.op.Matmul("", reshape_580, transpose_587).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::key")
        transpose_587._value = None
        reshape_589 = pybuda.op.Reshape("", matmul_588, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::key")
        matmul_588._value = None
        add_590 = pybuda.op.Add("", reshape_589, self.get_parameter("vit.encoder.layer.7.attention.attention.key.bias"))
        reshape_589._value = None
        hslice_591 = pybuda.op.HSlice("", add_590, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        add_590._value = None
        reshape_592 = pybuda.op.Reshape("", hslice_591, shape=(12, 197, 64)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        hslice_591._value = None
        transpose_593 = pybuda.op.Transpose("", reshape_592, dim0=-2, dim1=-1, out_dtype=torch.float32)
        reshape_592._value = None
        matmul_594 = pybuda.op.Matmul("", reshape_586, transpose_593).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        reshape_586._value = None
        transpose_593._value = None
        reshape_595 = pybuda.op.Reshape("", matmul_594, shape=(1, 12, 197, 197)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        matmul_594._value = None
        multiply_597 = pybuda.op.Multiply("", reshape_595, self.get_constant("const_70")).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        reshape_595._value = None
        softmax_598 = pybuda.op.Softmax("", multiply_597, dim=-1).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        multiply_597._value = None
        dropout_599 = pybuda.op.Identity("", softmax_598).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.dropout.Dropout::dropout")
        softmax_598._value = None
        reshape_600 = pybuda.op.Reshape("", dropout_599, shape=(12, 197, 197)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        dropout_599._value = None
        transpose_601 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.7.attention.attention.value.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_602 = pybuda.op.Matmul("", reshape_580, transpose_601).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::value")
        reshape_580._value = None
        transpose_601._value = None
        reshape_603 = pybuda.op.Reshape("", matmul_602, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::value")
        matmul_602._value = None
        add_604 = pybuda.op.Add("", reshape_603, self.get_parameter("vit.encoder.layer.7.attention.attention.value.bias"))
        reshape_603._value = None
        hslice_605 = pybuda.op.HSlice("", add_604, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        add_604._value = None
        transpose_606 = pybuda.op.Transpose("", hslice_605, dim0=-2, dim1=-1, out_dtype=torch.float32).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        hslice_605._value = None
        reshape_607 = pybuda.op.Reshape("", transpose_606, shape=(12, 64, 197)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        transpose_606._value = None
        transpose_608 = pybuda.op.Transpose("", reshape_607, dim0=-2, dim1=-1, out_dtype=torch.float32)
        reshape_607._value = None
        matmul_609 = pybuda.op.Matmul("", reshape_600, transpose_608).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        reshape_600._value = None
        transpose_608._value = None
        reshape_610 = pybuda.op.Reshape("", matmul_609, shape=(1, 12, 197, 64)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        matmul_609._value = None
        hstack_611 = pybuda.op.HStack("", reshape_610, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.linear.Linear::dense")
        reshape_610._value = None
        transpose_612 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.7.attention.output.dense.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_613 = pybuda.op.Matmul("", hstack_611, transpose_612).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.linear.Linear::dense")
        hstack_611._value = None
        transpose_612._value = None
        reshape_614 = pybuda.op.Reshape("", matmul_613, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.linear.Linear::dense")
        matmul_613._value = None
        add_615 = pybuda.op.Add("", reshape_614, self.get_parameter("vit.encoder.layer.7.attention.output.dense.bias"))
        reshape_614._value = None
        dropout_616 = pybuda.op.Identity("", add_615).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.dropout.Dropout::dropout")
        add_615._value = None
        add_617 = pybuda.op.Add("", dropout_616, add_578).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7")
        dropout_616._value = None
        add_578._value = None
        layernorm_618 = pybuda.op.Layernorm("", add_617, self.get_parameter("vit.encoder.layer.7.layernorm_after.weight"), self.get_parameter("vit.encoder.layer.7.layernorm_after.bias"), dim=-1, epsilon=0.0).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7/torch.nn.modules.normalization.LayerNorm::layernorm_after")
        reshape_619 = pybuda.op.Reshape("", layernorm_618, shape=(197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/torch.nn.modules.linear.Linear::dense")
        layernorm_618._value = None
        transpose_620 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.7.intermediate.dense.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_621 = pybuda.op.Matmul("", reshape_619, transpose_620).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/torch.nn.modules.linear.Linear::dense")
        reshape_619._value = None
        transpose_620._value = None
        reshape_622 = pybuda.op.Reshape("", matmul_621, shape=(1, 197, 3072)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/torch.nn.modules.linear.Linear::dense")
        matmul_621._value = None
        add_623 = pybuda.op.Add("", reshape_622, self.get_parameter("vit.encoder.layer.7.intermediate.dense.bias"))
        reshape_622._value = None
        gelu_624 = pybuda.op.Gelu("", add_623, approximate="none").set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/transformers.activations.GELUActivation::intermediate_act_fn")
        add_623._value = None
        reshape_625 = pybuda.op.Reshape("", gelu_624, shape=(197, 3072)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.linear.Linear::dense")
        gelu_624._value = None
        transpose_626 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.7.output.dense.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_627 = pybuda.op.Matmul("", reshape_625, transpose_626).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.linear.Linear::dense")
        reshape_625._value = None
        transpose_626._value = None
        reshape_628 = pybuda.op.Reshape("", matmul_627, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.linear.Linear::dense")
        matmul_627._value = None
        add_629 = pybuda.op.Add("", reshape_628, self.get_parameter("vit.encoder.layer.7.output.dense.bias"))
        reshape_628._value = None
        dropout_630 = pybuda.op.Identity("", add_629).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.dropout.Dropout::dropout")
        add_629._value = None
        add_631 = pybuda.op.Add("", dropout_630, add_617).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.7/transformers.models.vit.modeling_vit.ViTOutput::output")
        dropout_630._value = None
        add_617._value = None
        layernorm_632 = pybuda.op.Layernorm("", add_631, self.get_parameter("vit.encoder.layer.8.layernorm_before.weight"), self.get_parameter("vit.encoder.layer.8.layernorm_before.bias"), dim=-1, epsilon=0.0).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8/torch.nn.modules.normalization.LayerNorm::layernorm_before")
        reshape_633 = pybuda.op.Reshape("", layernorm_632, shape=(197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::query")
        layernorm_632._value = None
        transpose_634 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.8.attention.attention.query.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_635 = pybuda.op.Matmul("", reshape_633, transpose_634).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::query")
        transpose_634._value = None
        reshape_636 = pybuda.op.Reshape("", matmul_635, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::query")
        matmul_635._value = None
        add_637 = pybuda.op.Add("", reshape_636, self.get_parameter("vit.encoder.layer.8.attention.attention.query.bias"))
        reshape_636._value = None
        hslice_638 = pybuda.op.HSlice("", add_637, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        add_637._value = None
        reshape_639 = pybuda.op.Reshape("", hslice_638, shape=(12, 197, 64)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        hslice_638._value = None
        transpose_640 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.8.attention.attention.key.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_641 = pybuda.op.Matmul("", reshape_633, transpose_640).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::key")
        transpose_640._value = None
        reshape_642 = pybuda.op.Reshape("", matmul_641, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::key")
        matmul_641._value = None
        add_643 = pybuda.op.Add("", reshape_642, self.get_parameter("vit.encoder.layer.8.attention.attention.key.bias"))
        reshape_642._value = None
        hslice_644 = pybuda.op.HSlice("", add_643, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        add_643._value = None
        reshape_645 = pybuda.op.Reshape("", hslice_644, shape=(12, 197, 64)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        hslice_644._value = None
        transpose_646 = pybuda.op.Transpose("", reshape_645, dim0=-2, dim1=-1, out_dtype=torch.float32)
        reshape_645._value = None
        matmul_647 = pybuda.op.Matmul("", reshape_639, transpose_646).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        reshape_639._value = None
        transpose_646._value = None
        reshape_648 = pybuda.op.Reshape("", matmul_647, shape=(1, 12, 197, 197)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        matmul_647._value = None
        multiply_650 = pybuda.op.Multiply("", reshape_648, self.get_constant("const_80")).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        reshape_648._value = None
        softmax_651 = pybuda.op.Softmax("", multiply_650, dim=-1).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        multiply_650._value = None
        dropout_652 = pybuda.op.Identity("", softmax_651).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.dropout.Dropout::dropout")
        softmax_651._value = None
        reshape_653 = pybuda.op.Reshape("", dropout_652, shape=(12, 197, 197)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        dropout_652._value = None
        transpose_654 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.8.attention.attention.value.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_655 = pybuda.op.Matmul("", reshape_633, transpose_654).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::value")
        reshape_633._value = None
        transpose_654._value = None
        reshape_656 = pybuda.op.Reshape("", matmul_655, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::value")
        matmul_655._value = None
        add_657 = pybuda.op.Add("", reshape_656, self.get_parameter("vit.encoder.layer.8.attention.attention.value.bias"))
        reshape_656._value = None
        hslice_658 = pybuda.op.HSlice("", add_657, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        add_657._value = None
        transpose_659 = pybuda.op.Transpose("", hslice_658, dim0=-2, dim1=-1, out_dtype=torch.float32).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        hslice_658._value = None
        reshape_660 = pybuda.op.Reshape("", transpose_659, shape=(12, 64, 197)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        transpose_659._value = None
        transpose_661 = pybuda.op.Transpose("", reshape_660, dim0=-2, dim1=-1, out_dtype=torch.float32)
        reshape_660._value = None
        matmul_662 = pybuda.op.Matmul("", reshape_653, transpose_661).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        reshape_653._value = None
        transpose_661._value = None
        reshape_663 = pybuda.op.Reshape("", matmul_662, shape=(1, 12, 197, 64)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        matmul_662._value = None
        hstack_664 = pybuda.op.HStack("", reshape_663, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.linear.Linear::dense")
        reshape_663._value = None
        transpose_665 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.8.attention.output.dense.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_666 = pybuda.op.Matmul("", hstack_664, transpose_665).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.linear.Linear::dense")
        hstack_664._value = None
        transpose_665._value = None
        reshape_667 = pybuda.op.Reshape("", matmul_666, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.linear.Linear::dense")
        matmul_666._value = None
        add_668 = pybuda.op.Add("", reshape_667, self.get_parameter("vit.encoder.layer.8.attention.output.dense.bias"))
        reshape_667._value = None
        dropout_669 = pybuda.op.Identity("", add_668).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.dropout.Dropout::dropout")
        add_668._value = None
        add_670 = pybuda.op.Add("", dropout_669, add_631).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8")
        dropout_669._value = None
        add_631._value = None
        layernorm_671 = pybuda.op.Layernorm("", add_670, self.get_parameter("vit.encoder.layer.8.layernorm_after.weight"), self.get_parameter("vit.encoder.layer.8.layernorm_after.bias"), dim=-1, epsilon=0.0).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8/torch.nn.modules.normalization.LayerNorm::layernorm_after")
        reshape_672 = pybuda.op.Reshape("", layernorm_671, shape=(197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/torch.nn.modules.linear.Linear::dense")
        layernorm_671._value = None
        transpose_673 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.8.intermediate.dense.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_674 = pybuda.op.Matmul("", reshape_672, transpose_673).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/torch.nn.modules.linear.Linear::dense")
        reshape_672._value = None
        transpose_673._value = None
        reshape_675 = pybuda.op.Reshape("", matmul_674, shape=(1, 197, 3072)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/torch.nn.modules.linear.Linear::dense")
        matmul_674._value = None
        add_676 = pybuda.op.Add("", reshape_675, self.get_parameter("vit.encoder.layer.8.intermediate.dense.bias"))
        reshape_675._value = None
        gelu_677 = pybuda.op.Gelu("", add_676, approximate="none").set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/transformers.activations.GELUActivation::intermediate_act_fn")
        add_676._value = None
        reshape_678 = pybuda.op.Reshape("", gelu_677, shape=(197, 3072)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.linear.Linear::dense")
        gelu_677._value = None
        transpose_679 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.8.output.dense.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_680 = pybuda.op.Matmul("", reshape_678, transpose_679).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.linear.Linear::dense")
        reshape_678._value = None
        transpose_679._value = None
        reshape_681 = pybuda.op.Reshape("", matmul_680, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.linear.Linear::dense")
        matmul_680._value = None
        add_682 = pybuda.op.Add("", reshape_681, self.get_parameter("vit.encoder.layer.8.output.dense.bias"))
        reshape_681._value = None
        dropout_683 = pybuda.op.Identity("", add_682).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.dropout.Dropout::dropout")
        add_682._value = None
        add_684 = pybuda.op.Add("", dropout_683, add_670).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.8/transformers.models.vit.modeling_vit.ViTOutput::output")
        dropout_683._value = None
        add_670._value = None
        layernorm_685 = pybuda.op.Layernorm("", add_684, self.get_parameter("vit.encoder.layer.9.layernorm_before.weight"), self.get_parameter("vit.encoder.layer.9.layernorm_before.bias"), dim=-1, epsilon=0.0).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9/torch.nn.modules.normalization.LayerNorm::layernorm_before")
        reshape_686 = pybuda.op.Reshape("", layernorm_685, shape=(197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::query")
        layernorm_685._value = None
        transpose_687 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.9.attention.attention.query.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_688 = pybuda.op.Matmul("", reshape_686, transpose_687).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::query")
        transpose_687._value = None
        reshape_689 = pybuda.op.Reshape("", matmul_688, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::query")
        matmul_688._value = None
        add_690 = pybuda.op.Add("", reshape_689, self.get_parameter("vit.encoder.layer.9.attention.attention.query.bias"))
        reshape_689._value = None
        hslice_691 = pybuda.op.HSlice("", add_690, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        add_690._value = None
        reshape_692 = pybuda.op.Reshape("", hslice_691, shape=(12, 197, 64)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        hslice_691._value = None
        transpose_693 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.9.attention.attention.key.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_694 = pybuda.op.Matmul("", reshape_686, transpose_693).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::key")
        transpose_693._value = None
        reshape_695 = pybuda.op.Reshape("", matmul_694, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::key")
        matmul_694._value = None
        add_696 = pybuda.op.Add("", reshape_695, self.get_parameter("vit.encoder.layer.9.attention.attention.key.bias"))
        reshape_695._value = None
        hslice_697 = pybuda.op.HSlice("", add_696, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        add_696._value = None
        reshape_698 = pybuda.op.Reshape("", hslice_697, shape=(12, 197, 64)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        hslice_697._value = None
        transpose_699 = pybuda.op.Transpose("", reshape_698, dim0=-2, dim1=-1, out_dtype=torch.float32)
        reshape_698._value = None
        matmul_700 = pybuda.op.Matmul("", reshape_692, transpose_699).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        reshape_692._value = None
        transpose_699._value = None
        reshape_701 = pybuda.op.Reshape("", matmul_700, shape=(1, 12, 197, 197)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        matmul_700._value = None
        multiply_703 = pybuda.op.Multiply("", reshape_701, self.get_constant("const_90")).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        reshape_701._value = None
        softmax_704 = pybuda.op.Softmax("", multiply_703, dim=-1).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        multiply_703._value = None
        dropout_705 = pybuda.op.Identity("", softmax_704).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.dropout.Dropout::dropout")
        softmax_704._value = None
        reshape_706 = pybuda.op.Reshape("", dropout_705, shape=(12, 197, 197)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        dropout_705._value = None
        transpose_707 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.9.attention.attention.value.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_708 = pybuda.op.Matmul("", reshape_686, transpose_707).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::value")
        reshape_686._value = None
        transpose_707._value = None
        reshape_709 = pybuda.op.Reshape("", matmul_708, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::value")
        matmul_708._value = None
        add_710 = pybuda.op.Add("", reshape_709, self.get_parameter("vit.encoder.layer.9.attention.attention.value.bias"))
        reshape_709._value = None
        hslice_711 = pybuda.op.HSlice("", add_710, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        add_710._value = None
        transpose_712 = pybuda.op.Transpose("", hslice_711, dim0=-2, dim1=-1, out_dtype=torch.float32).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        hslice_711._value = None
        reshape_713 = pybuda.op.Reshape("", transpose_712, shape=(12, 64, 197)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        transpose_712._value = None
        transpose_714 = pybuda.op.Transpose("", reshape_713, dim0=-2, dim1=-1, out_dtype=torch.float32)
        reshape_713._value = None
        matmul_715 = pybuda.op.Matmul("", reshape_706, transpose_714).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        reshape_706._value = None
        transpose_714._value = None
        reshape_716 = pybuda.op.Reshape("", matmul_715, shape=(1, 12, 197, 64)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        matmul_715._value = None
        hstack_717 = pybuda.op.HStack("", reshape_716, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.linear.Linear::dense")
        reshape_716._value = None
        transpose_718 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.9.attention.output.dense.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_719 = pybuda.op.Matmul("", hstack_717, transpose_718).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.linear.Linear::dense")
        hstack_717._value = None
        transpose_718._value = None
        reshape_720 = pybuda.op.Reshape("", matmul_719, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.linear.Linear::dense")
        matmul_719._value = None
        add_721 = pybuda.op.Add("", reshape_720, self.get_parameter("vit.encoder.layer.9.attention.output.dense.bias"))
        reshape_720._value = None
        dropout_722 = pybuda.op.Identity("", add_721).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.dropout.Dropout::dropout")
        add_721._value = None
        add_723 = pybuda.op.Add("", dropout_722, add_684).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9")
        dropout_722._value = None
        add_684._value = None
        layernorm_724 = pybuda.op.Layernorm("", add_723, self.get_parameter("vit.encoder.layer.9.layernorm_after.weight"), self.get_parameter("vit.encoder.layer.9.layernorm_after.bias"), dim=-1, epsilon=0.0).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9/torch.nn.modules.normalization.LayerNorm::layernorm_after")
        reshape_725 = pybuda.op.Reshape("", layernorm_724, shape=(197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/torch.nn.modules.linear.Linear::dense")
        layernorm_724._value = None
        transpose_726 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.9.intermediate.dense.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_727 = pybuda.op.Matmul("", reshape_725, transpose_726).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/torch.nn.modules.linear.Linear::dense")
        reshape_725._value = None
        transpose_726._value = None
        reshape_728 = pybuda.op.Reshape("", matmul_727, shape=(1, 197, 3072)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/torch.nn.modules.linear.Linear::dense")
        matmul_727._value = None
        add_729 = pybuda.op.Add("", reshape_728, self.get_parameter("vit.encoder.layer.9.intermediate.dense.bias"))
        reshape_728._value = None
        gelu_730 = pybuda.op.Gelu("", add_729, approximate="none").set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/transformers.activations.GELUActivation::intermediate_act_fn")
        add_729._value = None
        reshape_731 = pybuda.op.Reshape("", gelu_730, shape=(197, 3072)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.linear.Linear::dense")
        gelu_730._value = None
        transpose_732 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.9.output.dense.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_733 = pybuda.op.Matmul("", reshape_731, transpose_732).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.linear.Linear::dense")
        reshape_731._value = None
        transpose_732._value = None
        reshape_734 = pybuda.op.Reshape("", matmul_733, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.linear.Linear::dense")
        matmul_733._value = None
        add_735 = pybuda.op.Add("", reshape_734, self.get_parameter("vit.encoder.layer.9.output.dense.bias"))
        reshape_734._value = None
        dropout_736 = pybuda.op.Identity("", add_735).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.dropout.Dropout::dropout")
        add_735._value = None
        add_737 = pybuda.op.Add("", dropout_736, add_723).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.9/transformers.models.vit.modeling_vit.ViTOutput::output")
        dropout_736._value = None
        add_723._value = None
        layernorm_738 = pybuda.op.Layernorm("", add_737, self.get_parameter("vit.encoder.layer.10.layernorm_before.weight"), self.get_parameter("vit.encoder.layer.10.layernorm_before.bias"), dim=-1, epsilon=0.0).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10/torch.nn.modules.normalization.LayerNorm::layernorm_before")
        reshape_739 = pybuda.op.Reshape("", layernorm_738, shape=(197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::query")
        layernorm_738._value = None
        transpose_740 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.10.attention.attention.query.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_741 = pybuda.op.Matmul("", reshape_739, transpose_740).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::query")
        transpose_740._value = None
        reshape_742 = pybuda.op.Reshape("", matmul_741, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::query")
        matmul_741._value = None
        add_743 = pybuda.op.Add("", reshape_742, self.get_parameter("vit.encoder.layer.10.attention.attention.query.bias"))
        reshape_742._value = None
        hslice_744 = pybuda.op.HSlice("", add_743, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        add_743._value = None
        reshape_745 = pybuda.op.Reshape("", hslice_744, shape=(12, 197, 64)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        hslice_744._value = None
        transpose_746 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.10.attention.attention.key.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_747 = pybuda.op.Matmul("", reshape_739, transpose_746).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::key")
        transpose_746._value = None
        reshape_748 = pybuda.op.Reshape("", matmul_747, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::key")
        matmul_747._value = None
        add_749 = pybuda.op.Add("", reshape_748, self.get_parameter("vit.encoder.layer.10.attention.attention.key.bias"))
        reshape_748._value = None
        hslice_750 = pybuda.op.HSlice("", add_749, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        add_749._value = None
        reshape_751 = pybuda.op.Reshape("", hslice_750, shape=(12, 197, 64)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        hslice_750._value = None
        transpose_752 = pybuda.op.Transpose("", reshape_751, dim0=-2, dim1=-1, out_dtype=torch.float32)
        reshape_751._value = None
        matmul_753 = pybuda.op.Matmul("", reshape_745, transpose_752).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        reshape_745._value = None
        transpose_752._value = None
        reshape_754 = pybuda.op.Reshape("", matmul_753, shape=(1, 12, 197, 197)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        matmul_753._value = None
        multiply_756 = pybuda.op.Multiply("", reshape_754, self.get_constant("const_100")).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        reshape_754._value = None
        softmax_757 = pybuda.op.Softmax("", multiply_756, dim=-1).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        multiply_756._value = None
        dropout_758 = pybuda.op.Identity("", softmax_757).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.dropout.Dropout::dropout")
        softmax_757._value = None
        reshape_759 = pybuda.op.Reshape("", dropout_758, shape=(12, 197, 197)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        dropout_758._value = None
        transpose_760 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.10.attention.attention.value.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_761 = pybuda.op.Matmul("", reshape_739, transpose_760).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::value")
        reshape_739._value = None
        transpose_760._value = None
        reshape_762 = pybuda.op.Reshape("", matmul_761, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::value")
        matmul_761._value = None
        add_763 = pybuda.op.Add("", reshape_762, self.get_parameter("vit.encoder.layer.10.attention.attention.value.bias"))
        reshape_762._value = None
        hslice_764 = pybuda.op.HSlice("", add_763, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        add_763._value = None
        transpose_765 = pybuda.op.Transpose("", hslice_764, dim0=-2, dim1=-1, out_dtype=torch.float32).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        hslice_764._value = None
        reshape_766 = pybuda.op.Reshape("", transpose_765, shape=(12, 64, 197)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        transpose_765._value = None
        transpose_767 = pybuda.op.Transpose("", reshape_766, dim0=-2, dim1=-1, out_dtype=torch.float32)
        reshape_766._value = None
        matmul_768 = pybuda.op.Matmul("", reshape_759, transpose_767).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        reshape_759._value = None
        transpose_767._value = None
        reshape_769 = pybuda.op.Reshape("", matmul_768, shape=(1, 12, 197, 64)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        matmul_768._value = None
        hstack_770 = pybuda.op.HStack("", reshape_769, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.linear.Linear::dense")
        reshape_769._value = None
        transpose_771 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.10.attention.output.dense.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_772 = pybuda.op.Matmul("", hstack_770, transpose_771).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.linear.Linear::dense")
        hstack_770._value = None
        transpose_771._value = None
        reshape_773 = pybuda.op.Reshape("", matmul_772, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.linear.Linear::dense")
        matmul_772._value = None
        add_774 = pybuda.op.Add("", reshape_773, self.get_parameter("vit.encoder.layer.10.attention.output.dense.bias"))
        reshape_773._value = None
        dropout_775 = pybuda.op.Identity("", add_774).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.dropout.Dropout::dropout")
        add_774._value = None
        add_776 = pybuda.op.Add("", dropout_775, add_737).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10")
        dropout_775._value = None
        add_737._value = None
        layernorm_777 = pybuda.op.Layernorm("", add_776, self.get_parameter("vit.encoder.layer.10.layernorm_after.weight"), self.get_parameter("vit.encoder.layer.10.layernorm_after.bias"), dim=-1, epsilon=0.0).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10/torch.nn.modules.normalization.LayerNorm::layernorm_after")
        reshape_778 = pybuda.op.Reshape("", layernorm_777, shape=(197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/torch.nn.modules.linear.Linear::dense")
        layernorm_777._value = None
        transpose_779 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.10.intermediate.dense.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_780 = pybuda.op.Matmul("", reshape_778, transpose_779).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/torch.nn.modules.linear.Linear::dense")
        reshape_778._value = None
        transpose_779._value = None
        reshape_781 = pybuda.op.Reshape("", matmul_780, shape=(1, 197, 3072)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/torch.nn.modules.linear.Linear::dense")
        matmul_780._value = None
        add_782 = pybuda.op.Add("", reshape_781, self.get_parameter("vit.encoder.layer.10.intermediate.dense.bias"))
        reshape_781._value = None
        gelu_783 = pybuda.op.Gelu("", add_782, approximate="none").set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/transformers.activations.GELUActivation::intermediate_act_fn")
        add_782._value = None
        reshape_784 = pybuda.op.Reshape("", gelu_783, shape=(197, 3072)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.linear.Linear::dense")
        gelu_783._value = None
        transpose_785 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.10.output.dense.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_786 = pybuda.op.Matmul("", reshape_784, transpose_785).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.linear.Linear::dense")
        reshape_784._value = None
        transpose_785._value = None
        reshape_787 = pybuda.op.Reshape("", matmul_786, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.linear.Linear::dense")
        matmul_786._value = None
        add_788 = pybuda.op.Add("", reshape_787, self.get_parameter("vit.encoder.layer.10.output.dense.bias"))
        reshape_787._value = None
        dropout_789 = pybuda.op.Identity("", add_788).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.dropout.Dropout::dropout")
        add_788._value = None
        add_790 = pybuda.op.Add("", dropout_789, add_776).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.10/transformers.models.vit.modeling_vit.ViTOutput::output")
        dropout_789._value = None
        add_776._value = None
        layernorm_791 = pybuda.op.Layernorm("", add_790, self.get_parameter("vit.encoder.layer.11.layernorm_before.weight"), self.get_parameter("vit.encoder.layer.11.layernorm_before.bias"), dim=-1, epsilon=0.0).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11/torch.nn.modules.normalization.LayerNorm::layernorm_before")
        reshape_792 = pybuda.op.Reshape("", layernorm_791, shape=(197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::query")
        layernorm_791._value = None
        transpose_793 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.11.attention.attention.query.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_794 = pybuda.op.Matmul("", reshape_792, transpose_793).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::query")
        transpose_793._value = None
        reshape_795 = pybuda.op.Reshape("", matmul_794, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::query")
        matmul_794._value = None
        add_796 = pybuda.op.Add("", reshape_795, self.get_parameter("vit.encoder.layer.11.attention.attention.query.bias"))
        reshape_795._value = None
        hslice_797 = pybuda.op.HSlice("", add_796, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        add_796._value = None
        reshape_798 = pybuda.op.Reshape("", hslice_797, shape=(12, 197, 64)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        hslice_797._value = None
        transpose_799 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.11.attention.attention.key.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_800 = pybuda.op.Matmul("", reshape_792, transpose_799).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::key")
        transpose_799._value = None
        reshape_801 = pybuda.op.Reshape("", matmul_800, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::key")
        matmul_800._value = None
        add_802 = pybuda.op.Add("", reshape_801, self.get_parameter("vit.encoder.layer.11.attention.attention.key.bias"))
        reshape_801._value = None
        hslice_803 = pybuda.op.HSlice("", add_802, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        add_802._value = None
        reshape_804 = pybuda.op.Reshape("", hslice_803, shape=(12, 197, 64)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        hslice_803._value = None
        transpose_805 = pybuda.op.Transpose("", reshape_804, dim0=-2, dim1=-1, out_dtype=torch.float32)
        reshape_804._value = None
        matmul_806 = pybuda.op.Matmul("", reshape_798, transpose_805).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        reshape_798._value = None
        transpose_805._value = None
        reshape_807 = pybuda.op.Reshape("", matmul_806, shape=(1, 12, 197, 197)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        matmul_806._value = None
        multiply_809 = pybuda.op.Multiply("", reshape_807, self.get_constant("const_110")).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        reshape_807._value = None
        softmax_810 = pybuda.op.Softmax("", multiply_809, dim=-1).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        multiply_809._value = None
        dropout_811 = pybuda.op.Identity("", softmax_810).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.dropout.Dropout::dropout")
        softmax_810._value = None
        reshape_812 = pybuda.op.Reshape("", dropout_811, shape=(12, 197, 197)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        dropout_811._value = None
        transpose_813 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.11.attention.attention.value.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_814 = pybuda.op.Matmul("", reshape_792, transpose_813).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::value")
        reshape_792._value = None
        transpose_813._value = None
        reshape_815 = pybuda.op.Reshape("", matmul_814, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention/torch.nn.modules.linear.Linear::value")
        matmul_814._value = None
        add_816 = pybuda.op.Add("", reshape_815, self.get_parameter("vit.encoder.layer.11.attention.attention.value.bias"))
        reshape_815._value = None
        hslice_817 = pybuda.op.HSlice("", add_816, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        add_816._value = None
        transpose_818 = pybuda.op.Transpose("", hslice_817, dim0=-2, dim1=-1, out_dtype=torch.float32).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        hslice_817._value = None
        reshape_819 = pybuda.op.Reshape("", transpose_818, shape=(12, 64, 197)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        transpose_818._value = None
        transpose_820 = pybuda.op.Transpose("", reshape_819, dim0=-2, dim1=-1, out_dtype=torch.float32)
        reshape_819._value = None
        matmul_821 = pybuda.op.Matmul("", reshape_812, transpose_820).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        reshape_812._value = None
        transpose_820._value = None
        reshape_822 = pybuda.op.Reshape("", matmul_821, shape=(1, 12, 197, 64)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfAttention::attention")
        matmul_821._value = None
        hstack_823 = pybuda.op.HStack("", reshape_822, slices=12).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.linear.Linear::dense")
        reshape_822._value = None
        transpose_824 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.11.attention.output.dense.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_825 = pybuda.op.Matmul("", hstack_823, transpose_824).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.linear.Linear::dense")
        hstack_823._value = None
        transpose_824._value = None
        reshape_826 = pybuda.op.Reshape("", matmul_825, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.linear.Linear::dense")
        matmul_825._value = None
        add_827 = pybuda.op.Add("", reshape_826, self.get_parameter("vit.encoder.layer.11.attention.output.dense.bias"))
        reshape_826._value = None
        dropout_828 = pybuda.op.Identity("", add_827).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11/transformers.models.vit.modeling_vit.ViTAttention::attention/transformers.models.vit.modeling_vit.ViTSelfOutput::output/torch.nn.modules.dropout.Dropout::dropout")
        add_827._value = None
        add_829 = pybuda.op.Add("", dropout_828, add_790).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11")
        dropout_828._value = None
        add_790._value = None
        layernorm_830 = pybuda.op.Layernorm("", add_829, self.get_parameter("vit.encoder.layer.11.layernorm_after.weight"), self.get_parameter("vit.encoder.layer.11.layernorm_after.bias"), dim=-1, epsilon=0.0).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11/torch.nn.modules.normalization.LayerNorm::layernorm_after")
        reshape_831 = pybuda.op.Reshape("", layernorm_830, shape=(197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/torch.nn.modules.linear.Linear::dense")
        layernorm_830._value = None
        transpose_832 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.11.intermediate.dense.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_833 = pybuda.op.Matmul("", reshape_831, transpose_832).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/torch.nn.modules.linear.Linear::dense")
        reshape_831._value = None
        transpose_832._value = None
        reshape_834 = pybuda.op.Reshape("", matmul_833, shape=(1, 197, 3072)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/torch.nn.modules.linear.Linear::dense")
        matmul_833._value = None
        add_835 = pybuda.op.Add("", reshape_834, self.get_parameter("vit.encoder.layer.11.intermediate.dense.bias"))
        reshape_834._value = None
        gelu_836 = pybuda.op.Gelu("", add_835, approximate="none").set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11/transformers.models.vit.modeling_vit.ViTIntermediate::intermediate/transformers.activations.GELUActivation::intermediate_act_fn")
        add_835._value = None
        reshape_837 = pybuda.op.Reshape("", gelu_836, shape=(197, 3072)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.linear.Linear::dense")
        gelu_836._value = None
        transpose_838 = pybuda.op.Transpose("", self.get_parameter("vit.encoder.layer.11.output.dense.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_839 = pybuda.op.Matmul("", reshape_837, transpose_838).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.linear.Linear::dense")
        reshape_837._value = None
        transpose_838._value = None
        reshape_840 = pybuda.op.Reshape("", matmul_839, shape=(1, 197, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.linear.Linear::dense")
        matmul_839._value = None
        add_841 = pybuda.op.Add("", reshape_840, self.get_parameter("vit.encoder.layer.11.output.dense.bias"))
        reshape_840._value = None
        dropout_842 = pybuda.op.Identity("", add_841).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11/transformers.models.vit.modeling_vit.ViTOutput::output/torch.nn.modules.dropout.Dropout::dropout")
        add_841._value = None
        add_843 = pybuda.op.Add("", dropout_842, add_829).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/transformers.models.vit.modeling_vit.ViTEncoder::encoder/transformers.models.vit.modeling_vit.ViTLayer::layer.11/transformers.models.vit.modeling_vit.ViTOutput::output")
        dropout_842._value = None
        add_829._value = None
        layernorm_844 = pybuda.op.Layernorm("", add_843, self.get_parameter("vit.layernorm.weight"), self.get_parameter("vit.layernorm.bias"), dim=-1, epsilon=0.0).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/transformers.models.vit.modeling_vit.ViTModel::vit/torch.nn.modules.normalization.LayerNorm::layernorm")
        add_843._value = None
        index_845 = pybuda.op.Index("", layernorm_844, dim=-2, start=0, stop=1, stride=1)
        layernorm_844._value = None
        reshape_846 = pybuda.op.Reshape("", index_845, shape=(1, 768)).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::")
        index_845._value = None
        transpose_847 = pybuda.op.Transpose("", self.get_parameter("classifier.weight"), dim0=-2, dim1=-1, out_dtype=torch.float32)
        matmul_848 = pybuda.op.Matmul("", reshape_846, transpose_847).set_src_layer("transformers.models.vit.modeling_vit.ViTForImageClassification::/torch.nn.modules.linear.Linear::classifier")
        reshape_846._value = None
        transpose_847._value = None
        add_849 = pybuda.op.Add("", matmul_848, self.get_parameter("classifier.bias"))
        matmul_848._value = None
        return add_849

    def process_framework_parameters(self, model):
        named_parameters = dict(model.state_dict().items())
        serialized_params = torch.load("generated_modules/asl_model_params.pt")
        named_parameters.update(serialized_params)
        named_buffers = dict(model.named_buffers())
        named_parameters.update(named_buffers)
        for name, torch_param in named_parameters.items():
            # Replace infinities with relevant numbers
            if torch.any(torch.isinf(torch_param)):
                torch_param = torch.where(torch.isposinf(torch_param), torch.tensor(1e4, dtype=torch_param.dtype), torch_param)
                torch_param = torch.where(torch.isneginf(torch_param), torch.tensor(-1e4, dtype=torch_param.dtype), torch_param)
                logger.warning(f"Replacing -inf and inf values in tensor param: {name}")
            tensor = torch_param.data
            if name in self._parameters:
                tensor.requires_grad = torch.is_floating_point(tensor)
                self.set_parameter(name, tensor)
            elif name in self._constants:
                if torch.numel(tensor) == 1 and len(tensor.shape) == 0:
                    tensor = tensor.reshape((1, 1))
                tensor.requires_grad = False
                if not torch.is_floating_point(tensor):
                    tensor = tensor.float()
                self.set_constant(name, tensor)
            else:
                logger.warning(f"{name} not found in self._parameters and self._constants")

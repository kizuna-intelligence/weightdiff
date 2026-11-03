import json

import gguf
import numpy as np
import pytest
from safetensors.numpy import save_file


@pytest.fixture
def model_factory(tmp_path):
    def make(name, widths=(4, 4, 4), changed=None, extras=None, config=None, tokenizer=None, template=None, sharded=False):
        root = tmp_path / name
        root.mkdir()
        tensors = {'model.embed_tokens.weight': np.arange(16384, dtype=np.float32).reshape(4096, 4),
                   'model.norm.weight': np.ones(4, dtype=np.float32)}
        for i, width in enumerate(widths):
            tensors[f'model.layers.{i}.mlp.weight'] = np.ones((width, 4), dtype=np.float32)
        if changed:
            tensors[changed] = tensors[changed] + 1
        if extras:
            tensors.update(extras)
        if sharded:
            names = sorted(tensors)
            mapping = {}
            for i, part in enumerate([names[::2], names[1::2]]):
                filename = f'model-{i+1:05}-of-00002.safetensors'
                save_file({n: tensors[n] for n in part}, str(root / filename))
                mapping.update({n: filename for n in part})
            (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': mapping}))
        else:
            save_file(tensors, str(root / 'model.safetensors'))
        cfg = {'architectures': ['TinyModel'], 'model_type': 'tiny', 'num_hidden_layers': len(widths), 'vocab_size': 4096}
        cfg.update(config or {})
        (root / 'config.json').write_text(json.dumps(cfg))
        (root / 'tokenizer_config.json').write_text(json.dumps(tokenizer or {}))
        if template is not None:
            (root / 'chat_template.jinja').write_text(template)
        return root
    return make


@pytest.fixture
def model_pair(tmp_path):
    def make(transform, dtype='Q8_0', shape=(16, 256), count=1):
        rng = np.random.default_rng(20)
        folder = tmp_path / 'original'
        folder.mkdir()
        tensors = {}
        target = tmp_path / 'converted.gguf'
        writer = gguf.GGUFWriter(target, 'llama')
        writer.add_block_count(count)
        writer.add_embedding_length(shape[-1])
        qtype = gguf.GGMLQuantizationType[dtype]
        for i in range(count):
            name = f'model.layers.{i}.self_attn.q_proj.weight'
            values = rng.normal(0, .25, shape).astype(np.float32)
            tensors[name] = values
            changed = np.ascontiguousarray(transform(values, i), dtype=np.float32)
            packed = gguf.quants.quantize(changed, qtype)
            writer.add_tensor(f'blk.{i}.attn_q.weight', packed, raw_dtype=qtype)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        save_file(tensors, str(folder / 'model.safetensors'))
        (folder / 'config.json').write_text(json.dumps({'model_type':'llama', 'num_hidden_layers':count}))
        return str(folder), str(target)
    return make

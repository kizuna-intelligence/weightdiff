import json

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

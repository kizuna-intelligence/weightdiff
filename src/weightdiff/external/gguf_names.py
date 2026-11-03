"""Use gguf-py's architecture-specific name map; never guess unknown names."""
import gguf


def map_names(hf_names, gguf_names, architecture, n_blocks):
    arch = next((arch for arch, name in gguf.MODEL_ARCH_NAMES.items() if name == architecture), None)
    if arch is None:
        return {}
    mapping = gguf.get_tensor_name_map(arch, n_blocks)
    available, result, used = set(gguf_names), {}, set()
    for name in sorted(hf_names):
        # Transformers' multimodal wrapper is absent from gguf-py's language map.
        normalized = name.replace('model.language_model.', 'model.', 1) if name.startswith('model.language_model.') else name
        target = mapping.get_name(normalized, try_suffixes=('.weight', '.bias'))
        if target in available and target not in used:
            result[name] = target
            used.add(target)
    return result

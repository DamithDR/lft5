CONFIG = {
    'falcon': {
        'target_modules': ["query_key_value"]
    },
    'llama': {
        'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj"]
    },
    'mistral': {
        'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj"]
    }
}

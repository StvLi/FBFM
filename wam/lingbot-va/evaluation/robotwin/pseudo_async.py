def should_wait_for_async_result(first, step_count, infer_delay_steps):
    return (not first) and step_count < infer_delay_steps + 1


def should_receive_async_result(first, step_count, infer_delay_steps):
    return (not first) and step_count == infer_delay_steps + 1


def should_request_next_kv_cache(first, step_count, chunk_size, infer_delay_steps):
    if first:
        return step_count == chunk_size - infer_delay_steps
    return step_count == chunk_size

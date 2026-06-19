# uniko (Python bindings)

Async-first Python SDK for the [uniko](https://github.com/rustic-ai/uniko)
cognitive memory engine, built with [PyO3](https://pyo3.rs) and
[maturin](https://www.maturin.rs).

> **Status:** alpha, under active construction. The async surface (recall,
> answer, query, ingest, goals/tasks, and the Locy logic surface) is being
> built out phase by phase. Synchronous (`*_sync`) skins, full type stubs, and
> prebuilt wheels are not yet available.

## Building locally

```bash
# from bindings/uniko-py/
maturin develop
python -c "import uniko; print(uniko.__file__)"
```

A C/C++ toolchain and `protobuf-compiler` (`protoc`) must be on `PATH` — the
uniko stack statically links the ONNX runtime (via uni-db's `provider-onnx`)
and several native dependencies.

## Example

```python
import asyncio
import uniko


async def main() -> None:
    uni = await uniko.Uniko.in_memory()
    agent = uni.agent("assistant")
    session = agent.session("conversation-1")
    await session.observe(uniko.Turn("user", "I prefer tea over coffee."))
    bundle = await agent.recall("beverage preference")
    for item in bundle.items:
        print(item.content)


asyncio.run(main())
```

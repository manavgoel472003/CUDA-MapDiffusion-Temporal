import onnx

p = "model/mapdiffusion_routeA/mapdiffusion.head.onnx"
m = onnx.load(p)

producer = {}
for i, n in enumerate(m.graph.node):
    for o in n.output:
        producer[o] = (i, n)

def trace(tensor, depth=0, seen=None):
    if seen is None:
        seen = set()
    indent = "  " * depth

    if tensor in seen:
        print(indent, tensor, "(seen)")
        return
    seen.add(tensor)

    if tensor not in producer:
        print(indent, tensor, "<input/initializer>")
        return

    i, n = producer[tensor]
    print(indent + f"{tensor} <- [{i}] {n.name} {n.domain}::{n.op_type}")
    print(indent + "  inputs:", list(n.input))
    if depth < 6:
        for inp in n.input:
            trace(inp, depth + 1, seen)

for out in ["cls_logits", "line_preds", "2958", "2956"]:
    print("=" * 100)
    print("TRACE", out)
    trace(out)

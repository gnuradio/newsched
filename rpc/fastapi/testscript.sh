
# Create the flowgraph
curl -v -H "Content-Type: application/json" POST      -d '{"name": "Foo"}' http://127.0.0.1:8000/flowgraph/create


# Create Blocks
curl -v -H "Content-Type: application/json" POST      -d '{"name": "src", "module": "blocks", "id": "vector_source_c", "properties": {"data": [1,2,3,4,5], "repeat": false }}' http://127.0.0.1:8000/block/create

curl -v -H "Content-Type: application/json" POST      -d '{"name": "copy_0", "module": "blocks", "id": "copy", "properties": {"itemsize": 8}}' http://127.0.0.1:8000/block/create

curl -v -H "Content-Type: application/json" POST      -d '{"name": "copy_1", "module": "blocks", "id": "copy", "properties": {"itemsize": 8}}' http://127.0.0.1:8000/block/create

curl -v -H "Content-Type: application/json" POST      -d '{"name": "snk", "module": "blocks", "id": "vector_sink_c", "properties": {}}' http://127.0.0.1:8000/block/create


# Connect Blocks
curl -v -H "Content-Type: application/json" POST      -d '{"flowgraph": "Foo", "src": ["src",0], "dest": ["copy_0",0] }' http://127.0.0.1:8000/flowgraph/connect

curl -v -H "Content-Type: application/json" POST      -d '{"flowgraph": "Foo", "src": ["copy_0",0], "dest": ["copy_1",0] }' http://127.0.0.1:8000/flowgraph/connect

curl -v -H "Content-Type: application/json" POST      -d '{"flowgraph": "Foo", "src": ["copy_1",0], "dest": ["snk",0] }' http://127.0.0.1:8000/flowgraph/connect


# Start Flowgraph
curl -v -H "Content-Type: application/json" POST      -d '{"name": "Foo"}' http://127.0.0.1:8000/flowgraph/start

curl -v -H "Content-Type: application/json" POST      -d '{"name": "Foo"}' http://127.0.0.1:8000/flowgraph/wait



# Create the flowgraph
curl -v -H "Content-Type: application/json" -X POST http://127.0.0.1:8000/flowgraph/foo/create

# Create Blocks
curl -v -H "Content-Type: application/json" POST      -d '{"module": "blocks", "id": "vector_source_c", "properties": {"data": [1,2,3,4,5], "repeat": false }}' http://127.0.0.1:8000/block/src/create

curl -v -H "Content-Type: application/json" POST      -d '{"module": "blocks", "id": "copy", "properties": {"itemsize": 8}}' http://127.0.0.1:8000/block/copy_0/create

curl -v -H "Content-Type: application/json" POST      -d '{"module": "blocks", "id": "copy", "properties": {"itemsize": 8}}' http://127.0.0.1:8000/block/copy_1/create

curl -v -H "Content-Type: application/json" POST      -d '{"module": "blocks", "id": "vector_sink_c", "properties": {}}' http://127.0.0.1:8000/block/snk/create


# Connect Blocks
curl -v -H "Content-Type: application/json" POST      -d '{"src": ["src",0], "dest": ["copy_0",0] }' http://127.0.0.1:8000/flowgraph/foo/connect

curl -v -H "Content-Type: application/json" POST      -d '{"src": ["copy_0",0], "dest": ["copy_1",0] }' http://127.0.0.1:8000/flowgraph/foo/connect

curl -v -H "Content-Type: application/json" POST      -d '{"src": ["copy_1",0], "dest": ["snk",0] }' http://127.0.0.1:8000/flowgraph/foo/connect


# Start Flowgraph
curl -v -H "Content-Type: application/json" -X POST http://127.0.0.1:8000/flowgraph/foo/start

curl -v -H "Content-Type: application/json" -X POST http://127.0.0.1:8000/flowgraph/foo/wait


# Query the Vector Sink Data
curl -v -H "Content-Type: application/json" POST      -d '{}' http://127.0.0.1:8000/block/snk/data

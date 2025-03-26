from flask import Flask, request, jsonify
from flask_cors import CORS
import grpc
import sign_data_pb2
import sign_data_pb2_grpc

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST'])
def hello_world():
    data = request.get_json()  # Ensure JSON parsing
    query = data.get("query", "Default message")  # Extract 'query' safely
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = sign_data_pb2_grpc.StreamDataServiceStub(channel)
    
        # Send the query as a string in RequestMessage
        request_message = sign_data_pb2.RequestMessage(data=query)
        response = stub.biDirectionalStream(request_message)
        return response.reply

if __name__ == '__main__':
    app.run(debug=True, port=8080)

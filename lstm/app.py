import grpc
import logging
import sign_data_pb2
import sign_data_pb2_grpc
from concurrent import futures

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

class StreamDataService(sign_data_pb2_grpc.StreamDataServiceServicer):
    def biDirectionalStream(self, request, context):
        logging.info(f"Received request data: {request.data}")
        response_text = request.data
        logging.info(f"Processed response: {response_text}")
        return sign_data_pb2.ResponseMessage(reply=response_text)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    sign_data_pb2_grpc.add_StreamDataServiceServicer_to_server(StreamDataService(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    logging.info("Server started on port 50051")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()

from concurrent import futures
from dotenv import load_dotenv
from model.refine_text import refine_text
import sys
import grpc
import logging
import sign_data_nlp_pb2
import sign_data_nlp_pb2_grpc

# Load environment variables from .env
load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout
)

MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024

class StreamDataService(sign_data_nlp_pb2_grpc.StreamDataServiceServicer):
    def biDirectionalStream(self, request, context):
        logging.info(f"Received request data: {request.data}")
        response_text = refine_text(request.data)
        logging.info(f"Processed response: {response_text}")
        return sign_data_nlp_pb2.ResponseMessage(reply=response_text)

def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH)
        ]
    )
    
    sign_data_nlp_pb2_grpc.add_StreamDataServiceServicer_to_server(StreamDataService(), server)
    server.add_insecure_port("[::]:50052")
    logging.info("Server started on port 50052")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()

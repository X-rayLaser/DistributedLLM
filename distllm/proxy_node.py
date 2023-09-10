import socketserver
from threading import Thread
from queue import Queue
from distllm import protocol

MAX_SIZE = 1

responses = Queue(MAX_SIZE)
requests = Queue(MAX_SIZE)


def run_proxy(host, client_port, node_port):
    thread = ServerThread(host, node_port)
    thread.start()

    with ThreadingTCPServer((host, client_port), FromClientToProxy) as server:
        server.serve_forever()

    print("Shuting down")
    thread.join()


class ServerThread(Thread):
    def __init__(self, host, port):
        super.__init__()
        self.host = host
        self.port = port

    def run(self):
        with ThreadingTCPServer((self.host, self.port), FromProxyToComputeNodeHandler) as server:
            server.serve_forever()


class FromProxyToComputeNodeHandler(socketserver.BaseRequestHandler):
    def handle(self):
        sock = self.request
        self.handshake(sock)

        try:
            self.exchange_messages_forever()
        except KeyboardInterrupt:
            print("Closing connection")

    def exchange_messages_forever(self):
        sock = self.request
        while True:
            message = requests.get()
            message.send(sock)
            
            msg, body = protocol.receive_message(sock)
            response = protocol.restore_message(msg, body)
            responses.put(response)

    def handshake(self):
        sock = self.request
        msg, body = protocol.receive_message(sock)

        message = protocol.restore_message(msg, body)
        if message == protocol.RequestGreeting():
            response = message
        else:
            response = protocol.ResponseWithError(operation=message.get_message(),
                                                  error='wrong_greeting', description='')
        
        response.send(sock)


class FromClientToProxy(socketserver.BaseRequestHandler):
    def handle(self):
        sock = self.request
        msg, body = protocol.receive_message(sock)
        message = protocol.restore_message(msg, body)
        print("Got message", msg)

        requests.put(message)

        response = responses.get()

        print("About to send message", response.get_message())
        response.send(sock)


class ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass

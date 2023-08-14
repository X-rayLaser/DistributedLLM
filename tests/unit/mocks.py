from distllm import protocol


class StableSocketMock:
    def __init__(self) -> None:
        self.data = bytes()
        self.step = 1
        self.idx = 0

    def recv(self, max_size):
        chunk = self.data[self.idx:self.idx + self.step]
        self.idx += self.step
        return chunk

    def sendall(self, buffer):
        self.data = buffer
        self.idx = 0


class VaryingChunkSocketMock(StableSocketMock):
    def __init__(self):
        super().__init__()
        self.num_reads = 0

    def recv(self, max_size):
        self.step = self.num_reads % 4
        self.num_reads += 1
        return super().recv(max_size)



class SimpleServerSocketMock(StableSocketMock):
    def __init__(self) -> None:
        super().__init__()
        self.message = None

    def sendall(self, buffer):
        """Pretend that this code executes by server which store response in data attribute"""
        msg = self.message.get_message()
        body = self.message.get_body()
        self.data = protocol.encode_message(msg, body)

    def set_reply_message(self, message):
        """Sets response that client will receive"""
        self.message = message


class ComplexServerSocketMock(StableSocketMock):
    def __init__(self):
        super().__init__()
        self.errors = {}
        self.responses = {}

        self.response_functions = {}

    def sendall(self, buffer):
        super().sendall(buffer)  # store data in the instance attributes

        # use subroutines to read data from socket (this instance) and decode client message
        msg, body = protocol.receive_message(self)
        message = protocol.restore_message(msg, body)
        message_text = message.get_message()

        # use appropriate response message with mocked body from the test code
        func = self.response_functions.get(message_text)
        if func:
            message_out = func()
            self.data = message_out.encode()
            self.idx = 0
            return

        error_body_out = self.errors.get(message_text)

        response_classes = {
            'slices_request': protocol.JsonResponseWithSlices,
            'status_request': protocol.JsonResponseWithStatus,
            'load_slice_request': protocol.JsonResponseWithLoadedSlice,
            'propagate_forward_request': protocol.ResponsePropagateForward,
            'request_file_submission_begin': protocol.ResponseFileSubmissionBegin,
            'request_submit_part': protocol.ResponseSubmitPart,
            'request_file_submission_end': protocol.ResponseFileSubmissionEnd
        }

        if error_body_out:
            message_out_class = protocol.ResponseWithError
            message_out = message_out_class(**error_body_out)
        else:
            try:
                message_out_class = response_classes[message_text]
            except KeyError:
                raise Exception(f'No response for message {message_text}')
            body_out = self.responses[message_text]
            message_out = message_out_class(**body_out)
        
        self.data = message_out.encode()
        self.idx = 0

    def set_error_body(self, msg, body):
        self.errors[msg] = body

    def set_reply_body(self, msg, body):
        self.responses[msg] = body

    def set_reply_function(self, msg, func):
        self.response_functions[msg] = func

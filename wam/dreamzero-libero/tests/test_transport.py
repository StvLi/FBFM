import socket

import numpy as np

from dreamzero_fbfm.transport import decode_array, encode_array, receive_message, send_message


def test_socket_round_trip_preserves_array():
    left, right = socket.socketpair()
    try:
        expected = np.arange(12, dtype=np.float32).reshape(3, 4)
        send_message(left, {"type": "array", "value": encode_array(expected)})
        message = receive_message(right)
        assert message is not None
        actual = decode_array(message["value"], dtype="float32")
        np.testing.assert_array_equal(actual, expected)
    finally:
        left.close()
        right.close()

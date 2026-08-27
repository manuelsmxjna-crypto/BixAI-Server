import io
import time

from PIL import Image
from starlette.responses import StreamingResponse

from app.main import STREAM_CHUNK_BYTES, _png_response, _stream_bytes


def test_stream_bytes_uses_bounded_chunks_and_closes_buffer():
    payload = b"x" * (STREAM_CHUNK_BYTES * 2 + 17)
    buffer = io.BytesIO(payload)

    chunks = list(_stream_bytes(buffer))

    assert [len(chunk) for chunk in chunks] == [STREAM_CHUNK_BYTES, STREAM_CHUNK_BYTES, 17]
    assert b"".join(chunks) == payload
    assert buffer.closed


def test_png_response_is_streamed_without_content_length():
    response = _png_response(Image.new("RGBA", (8, 8), (255, 0, 0, 128)), "test", time.perf_counter())

    assert isinstance(response, StreamingResponse)
    assert response.media_type == "image/png"
    assert "content-length" not in response.headers
    assert int(response.headers["x-bixai-output-bytes"]) > 0


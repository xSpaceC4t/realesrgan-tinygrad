import asyncio

async def send_tile(writer, obj):
    writer.write(len(obj).to_bytes(4, byteorder='big'))
    await writer.drain()
    writer.write(obj)
    await writer.drain()

async def recv_tile(reader):
    data = await reader.read(4)
    if not data:
        raise ConnectionError("Connection closed before header received")
    msg_len = int.from_bytes(data, byteorder='big')

    received_data = b''
    while len(received_data) < msg_len:
        chunk = await reader.read(msg_len - len(received_data))
        if not chunk:
            raise ConnectionError("Connection closed during data transmission")
        received_data += chunk

    return received_data
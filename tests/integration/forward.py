"""Minimal TCP forwarder: 0.0.0.0:8000 -> HOST:PORT. Lets the host reach the API without giving it outside access."""
import asyncio
import sys


async def pipe(reader, writer):
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except ConnectionError:
        pass
    finally:
        writer.close()


async def handle(client_reader, client_writer, host, port):
    try:
        upstream_reader, upstream_writer = await asyncio.open_connection(host, port)
    except OSError:
        client_writer.close()
        return
    await asyncio.gather(pipe(client_reader, upstream_writer), pipe(upstream_reader, client_writer))


async def main(host, port):
    server = await asyncio.start_server(lambda r, w: handle(r, w, host, port), '0.0.0.0', 8000)
    async with server:
        await server.serve_forever()


if __name__ == '__main__':
    asyncio.run(main(sys.argv[1], int(sys.argv[2])))

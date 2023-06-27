from typing import Generic, Optional, TypeVar


T = TypeVar('T')


class Server(Generic[T]):
    """
    Simple server model. Just stores a packet of type T and can be empty.
    """
    def __init__(self):
        self._packet: Optional[T] = None

    @property
    def ready(self) -> bool:
        """
        Check whether server is not serving any packet.
        """
        return self._packet is None

    @property
    def busy(self) -> bool:
        """
        Check whether server is serving a packet.
        """
        return self._packet is not None

    @property
    def size(self) -> int:
        """
        Get the number of packets being served (1 or 0)
        """
        return 1 if self._packet is not None else 0

    def pop(self) -> T:
        """
        Move from the server the packet being served.

        After this call busy server becomes ready, and the packet that was
        served is returned in the result.

        If the server was ready (empty), `RuntimeError` exception is thrown.

        Returns
        -------
        packet : T
            A packet that was served.
        """
        if self._packet is None:
            raise RuntimeError("attempted to pop from an empty server")
        packet = self._packet
        self._packet = None
        return packet

    def serve(self, packet: T) -> None:
        """
        Serve a new packet.

        If the server was ready, it starts serving the packet. That is,
        this packet is stored, server becomes busy and its size becomes equal
        one.

        If the server was already busy, `RuntimeError` is raised.

        Parameters
        ----------
        packet : T
        """
        if self._packet is not None:
            raise RuntimeError("attempted to put a packet to a busy server")
        self._packet = packet

    def __str__(self):
        suffix = "" if self._packet is None else f", packet={self._packet}"
        return f"(Server: busy={self.busy}{suffix})"

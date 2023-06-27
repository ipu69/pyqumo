from typing import TypeVar, Generic, Optional, List

import numpy as np

T = TypeVar('T')


class Queue(Generic[T]):
    """
    Abstract base class for the queues used in simulation models.

    Queues accept two methods:

    - `push(value: T) -> bool`
    - `pop() -> [T]`

    Push operation adds an item to the queue and returns true or false
    depending on whether the item was actually queued.

    Any queue will also implement four properties:

    - `capacity: int`
    - `size: int`
    - `empty: bool`
    - `full: bool`
    """
    @property
    def size(self) -> int:
        """
        Get the number of items in the queue.
        """
        raise NotImplementedError

    @property
    def capacity(self) -> int:
        """
        Get the maximum number of items in the queue.
        """
        raise NotImplementedError

    def push(self, item: T) -> bool:
        """
        Add an item to the queue.

        Parameters
        ----------
        item : T
            An item to add to the queue

        Returns
        -------
        success : bool
            True, if the item was really added.
        """
        raise NotImplementedError

    def pop(self) -> Optional[T]:
        """
        Extract an item from the queue.

        Returns
        -------
        item : T or None
            If queue failed to extract an item, it should return None
        """
        raise NotImplementedError

    def __len__(self):
        """
        Get the number of items in the queue (alias to size property).
        """
        return self.size

    @property
    def empty(self):
        """
        Check whether the queue is empty, i.e. number of items is zero.
        """
        return self.size == 0

    @property
    def full(self):
        """
        Check whether the queue is full, i.e. number of items equals capacity.
        """
        return self.size >= self.capacity


class FiniteFifoQueue(Queue[T]):
    """
    Finite queue representing a simple FIFO container.
    """

    def __init__(self, capacity: int):
        """
        Create a queue.

        Parameters
        ----------
        capacity : int
            Specifies maximum queue size
        """
        self.__items: List[T] = [None] * capacity
        self.__capacity = capacity
        self.__size = 0
        self.__head = 0
        self.__end = 0

    @property
    def capacity(self) -> int:
        return self.__capacity

    @property
    def size(self) -> int:
        return self.__size

    def push(self, item: T) -> bool:
        if self.full:
            return False
        self.__items[self.__end] = item
        self.__end = (self.__end + 1) % self.capacity
        self.__size += 1
        return True

    def pop(self) -> Optional[T]:
        if self.empty:
            return None
        item = self.__items[self.__head]
        self.__items[self.__head] = None
        self.__head = (self.__head + 1) % self.capacity
        self.__size -= 1
        return item

    def __repr__(self) -> str:
        """
        Get string representation of the queue.
        """
        if self.__head < self.__end:
            items = self.__items[self.__head:self.__end]
        elif self.__head >= self.__end and self.__size > 0:
            items = self.__items[self.__head:] + self.__items[:self.__end]
        else:
            items = []
        items_str = [str(item) for item in items]
        return f"(FiniteFifoQueue: q=[{', '.join(items_str)}], " \
               f"capacity={self.capacity}, size={self.size})"


class InfiniteFifoQueue(Queue[T]):
    """
    Infinite queue with FIFO order.
    """
    def __init__(self):
        self.__items = []

    @property
    def capacity(self):
        return np.inf

    @property
    def size(self):
        return len(self.__items)

    def push(self, item: T) -> bool:
        self.__items.append(item)
        return True

    def pop(self) -> Optional[T]:
        item: Optional[T] = None
        if len(self.__items) > 0:
            item = self.__items[0]
            self.__items = self.__items[1:]
        return item

    def __repr__(self):
        items = ', '.join([str(item) for item in self.__items])
        return f"(InfiniteFifoQueue: q=[{items}], size={self.size})"


"""How `Ros1Robot` reaches ROS: publish and subscribe JSON-shaped messages by topic.

`RosbridgeTransport` goes through a rosbridge websocket server with roslibpy, so the planner
machine needs no ROS install. Tests use an in-memory transport instead.
"""

from __future__ import annotations

from typing import Callable, Protocol


class Transport(Protocol):
    def subscribe(self, topic: str, message_type: str, callback: Callable[[dict], None]) -> None:
        """Call `callback` with each message (from another thread)."""
        ...

    def publish(self, topic: str, message_type: str, message: dict) -> None: ...

    def close(self) -> None: ...


class RosbridgeTransport:
    def __init__(self, host: str, port: int = 9090, timeout: float = 10.0):
        """Connect to a rosbridge websocket server (rosbridge_server's
        rosbridge_websocket.launch), waiting up to `timeout` s."""
        import roslibpy

        self._roslibpy = roslibpy
        self.ros = roslibpy.Ros(host=host, port=port)
        self.ros.run(timeout=timeout)
        self._publishers: dict[str, "roslibpy.Topic"] = {}
        self._subscribers: list["roslibpy.Topic"] = []

    def subscribe(self, topic: str, message_type: str, callback: Callable[[dict], None]) -> None:
        # queue_length 1: rosbridge drops all but the newest message for a slow client
        subscriber = self._roslibpy.Topic(self.ros, topic, message_type, queue_length=1)
        subscriber.subscribe(callback)
        self._subscribers.append(subscriber)

    def publish(self, topic: str, message_type: str, message: dict) -> None:
        publisher = self._publishers.get(topic)
        if publisher is None:
            publisher = self._roslibpy.Topic(self.ros, topic, message_type, queue_size=1)
            publisher.advertise()
            self._publishers[topic] = publisher
        publisher.publish(self._roslibpy.Message(message))

    def close(self) -> None:
        for subscriber in self._subscribers:
            subscriber.unsubscribe()
        for publisher in self._publishers.values():
            publisher.unadvertise()
        self.ros.terminate()

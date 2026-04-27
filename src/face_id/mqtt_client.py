from __future__ import annotations

import json

try:
    import paho.mqtt.client as mqtt
    _PAHO_AVAILABLE = True
except ImportError:
    _PAHO_AVAILABLE = False


class MqttNotifier:
    """Publishes face-recognition events to an MQTT broker."""

    def __init__(
        self,
        broker: str,
        port: int = 1883,
        topic_prefix: str = "face-id",
    ) -> None:
        if not _PAHO_AVAILABLE:
            raise RuntimeError(
                "paho-mqtt is not installed. Run: uv sync --extra mqtt"
            )
        self.broker = broker
        self.port = port
        self.topic_prefix = topic_prefix.rstrip("/")

        self._client = mqtt.Client()
        self._connected = False
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect

    # ------------------------------------------------------------------
    # Internal callbacks
    # ------------------------------------------------------------------

    def _on_connect(self, client, userdata, flags, rc) -> None:
        if rc == 0:
            self._connected = True
            print(f"MQTT connected → {self.broker}:{self.port}")
            self._publish("sensors", "off")
        else:
            print(f"MQTT connection failed (rc={rc})")

    def _on_disconnect(self, client, userdata, rc) -> None:
        self._connected = False
        if rc != 0:
            print(f"MQTT disconnected unexpectedly (rc={rc})")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def connect(self) -> None:
        try:
            self._client.connect(self.broker, self.port, keepalive=60)
            self._client.loop_start()
        except Exception as exc:
            print(f"MQTT connect error: {exc}")

    def publish_approved(self, name: str, similarity: float) -> None:
        payload = json.dumps({"person": name, "similarity": round(similarity, 2)})
        self._publish("approved", payload)

    def publish_rejected(self) -> None:
        self._publish("rejected", json.dumps({"person": "unknown"}))

    def disconnect(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()
        self._connected = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _publish(self, subtopic: str, payload: str) -> None:
        if not self._connected:
            return
        topic = f"{self.topic_prefix}/{subtopic}"
        self._client.publish(topic, payload)
        print(f"MQTT → {topic}: {payload}")

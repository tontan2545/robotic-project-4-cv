# listen to mqtt server on port 1883 using paho-mqtt
# and publish to the local mqtt server on port 188
import paho.mqtt.client as mqtt


def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe("test")


mqttc = mqtt.Client()
mqttc.on_connect = on_connect

mqttc.connect("localhost", 1883, 60)
mqttc.loop_forever()

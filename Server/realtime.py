import redis

redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

def post(stream, data):
    redis_client.xadd(stream, data)

def get(stream, count=10):
    messages = redis_client.xrange(stream, count=count)
    for message_id, message_data in messages:
        print("Message ID:", message_id.decode())
        print("Message Data:", message_data)

if __name__ == "__main__":
    stream = 'stream'

    for i in range(5):
        data = {"valeur": str(i)}
        post(stream, data)

    get(stream)


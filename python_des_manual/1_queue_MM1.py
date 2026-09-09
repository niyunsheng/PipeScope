import heapq
import random
import queue
from dataclasses import dataclass

class Event:
    def __init__(self, time, seq, callback, *callback_args):
        self.time = time
        self.seq = seq
        self.callback = callback
        self.callback_args = callback_args

    def __lt__(self, other):
        return (self.time, self.seq) < (other.time, other.seq)


class DES_Engine:
    def __init__(self,):
        self.event_list = []
        self.now = 0
        self.seq = 0

    def _finished(self, ):
        return len(self.event_list) == 0

    def run(self, until=None):
        while not self._finished():
            event_time = self._get_event_time()
            if until is not None and event_time >= until:
                break
            event = self._get_event()
            self.now = event.time
            if event.callback is not None:
                event.callback(*event.callback_args)

    def _get_event_time(self,):
        return self.event_list[0].time

    def _get_event(self,):
        return heapq.heappop(self.event_list)

    def add_event(self, event: Event):
        heapq.heappush(self.event_list, event)

    def schedule(self, time, callback, *args):
        assert time >= self.now
        self.seq += 1
        self.add_event(Event(time, self.seq, callback, *args))

    def reset(self):
        self.now = 0
        self.seq = 0
        self.event_list = []


enginer = DES_Engine()
_queue = queue.Queue()
last_time = 0
averate_n = 0
service_in_use = 0
remain_time = 0
service_n = 0

@dataclass
class Item:
    id: int
    arrive_time: float
    service_time: float


def arrive_and_in_queue(service_item):
    global averate_n, last_time, service_in_use, service_n
    averate_n += (enginer.now-last_time)*service_n
    service_in_use += (enginer.now-last_time)*(service_n>0)
    last_time = enginer.now

    service_n += 1
    _queue.put(service_item)
    if service_n <= 1:
        de_queue_and_service()

def de_queue_and_service(): 
    if _queue.empty():
        return
    service_item = _queue.get()
    enginer.schedule(enginer.now + service_item.service_time, end_service, service_item)

def end_service(service_item):
    global averate_n, last_time, service_in_use, service_n, remain_time
    averate_n += (enginer.now-last_time)*service_n
    service_in_use += (enginer.now-last_time)*(service_n>0)
    last_time = enginer.now
    remain_time += enginer.now - service_item.arrive_time

    service_n -= 1
    de_queue_and_service()

random.seed(0)
arrive_time = 0
def generate_custom(a, u):
    global arrive_time
    arrive_time += random.expovariate(a)
    service_time = random.expovariate(u)
    return arrive_time, service_time

n = 100000
a, u = 0.8, 1.0
for id in range(n):
    arrive_time, service_time = generate_custom(a, u)
    service_item = Item(id, arrive_time, service_time)
    enginer.schedule(arrive_time, arrive_and_in_queue, service_item)

enginer.run()

p = a/u
L = p/(1-p)
print("Theoretical value: ", p, L, 1/(u-a), 1/(u-a)*a)
print("Real:", service_in_use/enginer.now ,averate_n/enginer.now, remain_time/n, remain_time/n*a)

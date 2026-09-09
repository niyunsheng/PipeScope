import heapq


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

def tick():
    print("tick", enginer.now)
    enginer.schedule(enginer.now + 1, tick)

def pong(n):
    print("pong", enginer.now)
    if n > 1:
        enginer.schedule(enginer.now + 0.3, ping, n-1)

def ping(n):
    print("ping", enginer.now)
    if n > 1:
        enginer.schedule(enginer.now + 0.3, pong, n-1)

enginer.schedule(0, tick)
enginer.schedule(1, ping, 5)

enginer.run(until=10)

print("break")
enginer.run(until=15)

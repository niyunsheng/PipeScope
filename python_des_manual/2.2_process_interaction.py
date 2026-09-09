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

meet_latency = 0.5

def script_A():
    for i in range(3):
        waited = yield ("meet", "B", f"p{i}")
        if waited > 1:
            yield ("work", 2)
        else:
            yield ("work", 1)

def script_B():
    for routine in [("work", 2), ("work", 1), ("meet", "A", "p0"), ("work", 1), ("meet", "A", "p1"), ("work", 4), ("meet", "A", "p2"),]:
        yield routine

generator = {
    "A": script_A(),
    "B": script_B()
}

def resume(person, value=None):
    try:
        routine = generator[person].send(value)
    except StopIteration:
        routine = None
    if routine is not None:
        process_item(person, routine)

meet_keys = dict()


def work_done(person, work_start):
    print(f"{person} work from {work_start} to {enginer.now}")
    resume(person)

def meet_done(pA, pB, place, wait_start):
    print(f"{pA} wait {pB} @{place} from {wait_start} to {enginer.now-meet_latency}, meet {meet_latency}")
    wait_time = enginer.now-meet_latency-wait_start
    resume(pA, wait_time)
    resume(pB, 0)


def process_item(person, routine):

    action= routine[0]
    if action == "work":
        enginer.schedule(enginer.now + routine[1], work_done, person, enginer.now)
    elif action == "meet":
        meet_key = (routine[1], person, routine[2])
        if meet_key in meet_keys:
            wait_start = meet_keys[meet_key]
            del meet_keys[meet_key]
            enginer.schedule(enginer.now + meet_latency, meet_done, routine[1], person, routine[2], wait_start)
        else:
            meet_keys[(person, routine[1], routine[2])] = enginer.now


resume("A")
resume("B")

enginer.run()

print("Success" if len(meet_keys)==0 else "Fail")
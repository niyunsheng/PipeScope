# Hand-written discrete-event simulation: a problem set

Goal: build a pipeline-schedule simulator from scratch in plain Python 3, standard library only, without reading `src/sim/`. Work through the exercises in order; each has a **Task**, an **Acceptance** check you can run yourself, and **Hints** you should open only after you get stuck. The `.py` files in this directory are one person's solutions, kept for reference; write yours before looking.

Part 1 is a warm-up on generic discrete-event simulation (DES). It builds the "clock jumps to the next event" mental model and has you write all three classic engine styles by hand. Part 2 moves to pipelines; exercises to be added.

---

## Part 1: DES warm-up

### Exercise 0. Future event list

Reference solution: `0_event.py`

**Task.** Write a minimal DES kernel class with:

- `schedule(time, callback, *args)`: register that `callback(*args)` should run at `time`.
- `now`: the current simulation time.
- `run(until=None)`: repeatedly take the earliest registered event, move `now` to its time, and call its callback, until no events remain or the next event is at or beyond `until`.

Requirements:

1. Use `heapq` as the priority queue.
2. Two events registered for the same time must run in the order they were registered, deterministically.
3. A callback may call `schedule` itself. Scheduling into the past (time < `now`) is a bug; make it fail loudly.
4. Build two toys on the kernel: a clock that prints once every 1 time unit; and ping-pong, where A receives the ball at t and schedules B to receive it at t + 0.3, for a fixed number of rounds, then stops by itself without any external cut-off.

**Acceptance.**

- `run(until=10)` on the clock pops exactly 10 events.
- Ping-pong prints at 0, 0.3, 0.6, ... and stops on its own after the agreed number of hits.
- Put both toys into the same kernel and call `run` once; the interleaved output is in time order, and ties (e.g. the clock at 3 and a ping at 3.0) come out in registration order.

<details><summary>Hints</summary>

- `heapq` compares heap entries with `<`. Plain objects are not comparable; functions never are. Make the entry compare on `(time, seq)` where `seq` is a counter the engine increments on every `schedule`. Ask yourself why `seq` must be assigned by the engine and not passed in by the caller.
- Check the heap top before popping when implementing `until`, otherwise the first event beyond `until` is lost.
- 0.3 added ten times is not 3.0 in floating point. Decide now how you handle this (integer ticks, or `t0 + k * step`); it comes back in later exercises.
- Put the stop condition of ping-pong in the callback's arguments (a remaining-count), not in a global.

</details>

**Takeaways.** An event is an instant, not an interval; "something lasts d" is two events. Only work with time passing between now and it belongs on the heap; anything happening "right now" is a direct function call.

---

### Exercise 1. M/M/1 queue

Reference solution: `1_queue_MM1.py`

**Task.** Using the kernel from exercise 0, simulate a single-server FIFO queue with Poisson arrivals (rate λ) and exponential service times (rate μ). Use `random.expovariate` with a fixed seed. Measure:

- L, the time-average number of customers in the system;
- server utilization, the fraction of time the server is busy;
- W, the mean time a customer spends in the system.

Print them next to the theoretical values for ρ = λ/μ: L = ρ/(1−ρ), W = 1/(μ−λ), utilization = ρ. Also print both sides of Little's law, L = λW.

**Acceptance.**

- λ=0.8, μ=1, 10⁶ customers: all three statistics within 2% of theory.
- λ=0.95, same run length: the error is noticeably larger. Explain why in a comment.
- Little's law holds to within the same tolerance when λ is taken as the measured n / T, not the nominal rate.

<details><summary>Hints</summary>

- There are only two event types: arrival and service end. Everything else (enqueue, dequeue, start service) happens at the same instant as one of those and is a direct call.
- An arrival time is the cumulative sum of inter-arrival intervals. Scheduling every customer at its own interval puts all of them in the first few time units.
- The whole state is one integer n. Arrival: n += 1; if n became 1, start service. Service end: n −= 1; if n > 0, start the next.
- All three statistics are time-weighted: whenever n changes, add n × (time since the last change) to an accumulator. For W, keep a FIFO of arrival times.
- Why λ=0.95 converges slowly: the relaxation time of M/M/1 grows like 1/(μ(1−ρ)²), so consecutive samples are far more correlated and you have an order of magnitude fewer effectively independent ones.
- Keep model state (n, accumulators) out of the engine class; the engine will be reused unchanged later.

</details>

---

### Exercise 2. Program-driven warm-up: two people passing a baton

Reference solutions: `2.0_event_scheduling.py`, `2.1_activity_scanning.py`, `2.2_process_interaction.py`

Exercises 0 and 1 are **data-driven**: whoever is ready moves, and the order of events is an output of the simulation. Pipeline schedules are **program-driven**: each participant holds a fixed script, and the simulation only computes *when* each step happens. This exercise isolates the one hard part of program-driven simulation, "a blocked participant continues when its peer arrives", in a toy with no pipeline concepts, and has you implement it three ways.

**The toy.** Two people, A and B. Each has a script, a list of instructions of two kinds:

- `work(d)`: spend d time units working.
- `meet(peer, place)`: go to `place` and wait for `peer`. Whoever arrives first waits. Once both are there, the meeting completes `latency` time units later (a parameter, default 0) and both continue with their next instruction.

Example: A = `[work 1, meet B x, work 3, meet B y]`, B = `[work 2, meet A x, work 1, meet A y]`. With latency 0: x completes at 2 (A waited 1), y completes at 5 (B waited 2).

**Task.** Write three simulators for this toy. All three must:

- print, for each person, the start and end time of every instruction;
- print, for each meeting, who waited for whom, from when to when, and (if latency > 0) the latency as a separate segment;
- detect deadlock: if a `meet` is removed from one script, report which person is stuck at which place waiting for whom since what time, rather than looping forever or exiting silently;
- give identical output for the same script and latency.

**2.0 Event scheduling.** Reuse the kernel from exercise 0. `work` schedules its completion; `meet` must not consume simulation time by itself.

**2.1 Activity scanning.** No kernel, no heap, no global clock. Each person has a program counter and a *local* clock. A main loop sweeps over all people repeatedly; each person takes one step if it can. A full sweep with no progress while someone is still waiting is a deadlock.

**2.2 Process interaction.** Each person's script is a Python generator. It `yield`s requests such as `("work", d)` or `("meet", peer, place)`; the engine (the kernel from exercise 0 underneath) decides what to do with the request and resumes the generator with `send(result)` when the person may continue. Write at least one script that is real code, with a loop and a branch on the waiting time sent back, not just a list.

**Acceptance.**

- The example script gives x=2, y=5, A waits 1, B waits 2 in all three versions (latency 0). With latency 0.5: x=2.5, y=5.5.
- A second script of your own with three or more meetings, run through all three versions, gives identical per-instruction times.
- A script in which A arrives first at a meet but is swept *after* B by the 2.1 main loop. 2.1 must give the same times as 2.0.
- Removing one `meet` makes all three versions report a deadlock naming the stuck person, place, peer and time.

<details><summary>Hints for 2.0</summary>

- Only `work` needs the heap. At a `meet`, the first arriver records "I am at `place` waiting for `peer` since `now`" and then does nothing further: it schedules no event. The second arriver finds that record, computes the completion time, and schedules `meet_done` at `now + latency`. `meet_done` advances *both* people. With latency 0 this is a zero-delay event; keep it scheduled rather than special-casing it.
- Key the registry on `(waiter, peer, place)`; the arriver looks up `(peer, self, place)`. Remove the entry when paired.
- Deadlock = the heap is empty but the registry is not. The registry already holds everything the report needs.
- Success is not "the registry is empty"; also check that every script actually finished.

</details>

<details><summary>Hints for 2.1</summary>

- Do not use a generator for the script. When the peer has not arrived, the same `meet` instruction must be re-examined on the next sweep; a list with an explicit index can do that, `next()` cannot go back.
- `meet` only *registers* arrival (once). It never wakes anyone. The waiter discovers the peer's registration on a later sweep by itself.
- Do not delete a registration when the second person pairs with it: the first person still has to see it on their own sweep.
- **Registration order is not time order.** Because there is no global clock, the person swept first may have the larger clock. The meeting completes at `max(both arrival clocks) + latency`, and "who waited" is decided by comparing arrival clocks, never by who registered first. Getting this wrong makes a clock go backwards.
- If your `pc` advances before the instruction completes, make the finished-check also require "not blocked", or deadlock on the last instruction becomes a silent success.

</details>

<details><summary>Hints for 2.2</summary>

- Separate the script from the interpreter: the generator only yields requests; a `resume(person, value=None)` function does `gen.send(value)`, catches `StopIteration`, and dispatches the yielded request. The script must not call into the engine.
- `gen.send(None)` is equivalent to `next(gen)` and is the only legal first call.
- The value sent back is per-process: the waiter gets its waiting time, the late arriver gets 0.
- The only difference from 2.0 is where the program counter lives: in 2.0 it is an explicit index (or an external iterator), in 2.2 it is the generator's suspension point.

</details>

**Takeaways.** Three ways to organize a DES: event scheduling (a global heap, logic split across callbacks), activity scanning (per-participant local clocks, a sweep to a fixed point, no heap), process interaction (coroutines per participant on top of a heap). Only activity scanning needs no global clock; the price is that any decision depending on simulation time (e.g. "give up after waiting 3") must be justified by "the peer's clock has already passed t", which a global clock provides for free. `src/sim/engine.ts` in this repository is an activity-scanning engine.

#### Questions to answer after finishing 2.0, 2.1 and 2.2

- In each version, how does the first arriver learn that the peer has arrived? Which version needs an explicit wake-up and which does not, and why?
- In 2.0, why does `meet` stay off the heap while `work` goes on it? Is this the same criterion that made dequeue a direct call in exercise 1?
- In 2.1, A's clock can be at 5 while B's is still at 2. Can that happen in 2.0? Why does 2.1 tolerate this inconsistency and still get the right answer?
- With latency, the meeting completes at `max(both arrivals) + latency`. Which line implements this in each version? Is it the same formula as the rendezvous rule in `src/sim/engine.ts` (`start = max(sendPosted, recvPosted)`, `landed = start + latency`)?
- Add "give up if waited more than 3". How would each version implement it? In 2.0 and 2.2, what should the timeout event do when it fires after the meeting already happened? In 2.1, on what grounds can the waiter conclude the peer will be too late, and what happens to that reasoning when the peer is itself stuck?

---

## Part 2: Pipelines

To be added.

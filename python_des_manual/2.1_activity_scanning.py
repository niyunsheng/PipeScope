meet_latency = 0.5
persons = ["A", "B"]
routines = {
    "A": [("work", 5), ("meet", "B", "x"), ("work", 3), ("meet", "B", "y")],
    "B": [("work", 2), ("work", 1), ("meet", "A", "x"), ("work", 1), ("meet", "A", "y"), ("work", 2)]
}
pc = {
    p:0 for p in persons
}
clock = {
    p:0 for p in persons
}
meet_keys = {}
blocked = {
    p:None for p in persons
}
def step(person):
    if blocked[person] is not None:
        assert blocked[person] in meet_keys
        wait_time, end_time = meet_keys[blocked[person]]
        if end_time is not None:
            del meet_keys[blocked[person]]
            blocked[person] = None
            clock[person] = end_time
            return True
        return False
    if pc[person] >= len(routines[person]):
        return False

    routine = routines[person][pc[person]]
    pc[person] += 1
    action= routine[0]
    if action == "work":
        print(f"{person} work from {clock[person]} to {clock[person]+routine[1]}")
        clock[person] += routine[1]
        
    elif action == "meet":
        meet_key = (routine[1], person, routine[2])
        if meet_key in meet_keys:
            wait_start = meet_keys[meet_key][0]
            if wait_start < clock[person]:
                print(f"{routine[1]} wait {person} @{routine[2]} from {wait_start} to {clock[person]}")
            else:
                print(f"{person} wait {routine[1]} @{routine[2]} from {clock[person]} to {wait_start}")
            # clock[person] += meet_latency # Important Error
            clock[person] = max(clock[person], wait_start) + meet_latency
            meet_keys[meet_key] = (wait_start, clock[person])
        else:
            meet_keys[(person, routine[1], routine[2])] = (clock[person], None)
            blocked[person] = (person, routine[1], routine[2])
    return True

def all_finished():
    return all(pc[p]==len(routines[p]) and blocked[p] is None for p in persons)

while True:
    progress = False
    for person in persons:
        progress |= step(person)
    if all_finished():
        break
    if not progress:
        print("Fail")
        for (who, peer, place), (since, _) in meet_keys.items():
            print(f"Deadlock: {who} waiting for {peer} at {place} since t={since}")
        break

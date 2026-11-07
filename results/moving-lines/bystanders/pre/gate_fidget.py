"""Zero calls. How often does a fidgeting bystander pass CT3's join gate (src/sgi/resolve.py Resolver._moved: within
history_s = 20 s, walked moved_m = 1.0 m from the latest place, or came _STEP_UP_M = 0.55 m nearer the target)?
Fidget as the run describes it: every U(5, 10) s a new point uniform within 0.35 m of the home place (walks
there; taken as instant here). 92 s episodes, 20000 per bystander; checked at every fidget move."""
import math, random
TARGETS = {'cafe_line2': (5.0, -0.5), 'room_cart': (5.0, -1.0), 'sweet_spot_line2': (15.8, 5.2)}
BYS = {'cafe_line2': {'male_0': (4.2, -2.6), 'female_1': (5.85, -3.45), 'female_3': (6.6, -3.7)},
       'room_cart': {'male_2': (4.2, -4.35), 'female_1': (4.0, -5.15), 'male_3': (3.4, -5.5)},
       'sweet_spot_line2': {'male_0': (16.55, 3.1), 'female_1': (16.9, 2.2), 'female_3': (17.5, 2.6)}}
rng = random.Random(1)


def point(home):
    r, a = 0.35 * math.sqrt(rng.random()), rng.random() * 2 * math.pi
    return home[0] + r * math.cos(a), home[1] + r * math.sin(a)


for scene, bys in BYS.items():
    tgt = TARGETS[scene]
    for name, home in bys.items():
        ever = time_open = 0.0
        for _ in range(20000):
            t, hist, passed, open_s = 0.0, [(0.0, point(home))], False, 0.0
            while t < 92.0:
                dt = rng.uniform(5, 10)
                t += dt
                xy = point(home)
                hist = [h for h in hist if t - h[0] <= 20.0] + [(t, xy)]
                gap = math.dist(xy, tgt)
                ok = any(math.dist(xy, h[1]) >= 1.0 or math.dist(h[1], tgt) - gap >= 0.55 for h in hist[:-1])
                if ok:
                    passed = True
                    open_s += min(dt, 92.0 - (t - dt))
            ever += passed
            time_open += open_s / 92.0
        print(f'{scene:17} {name:9} passes the gate at least once in {ever / 20000:.0%} of episodes; '
              f'open about {time_open / 20000:.1%} of the time')

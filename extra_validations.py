import time
from model.strategy_optimizer.pit_stop_optimizer import optimize_strategy, print_strategy_report

scenarios = [
    {'circuit': 'Miami Grand Prix', 'laps': 57},
    {'circuit': 'Emilia Romagna Grand Prix', 'laps': 63},
    {'circuit': 'British Grand Prix', 'laps': 52},
    {'circuit': 'Italian Grand Prix', 'laps': 53}
]

for s in scenarios:
    c = s['circuit']
    l = s['laps']
    print(f"\n--- SCENARIO: {c} ---")
    t0 = time.time()
    res = optimize_strategy(
        driver_id=1, team_encoded=3, circuit=c,
        grid_position=5, total_laps=l, start_compound='MEDIUM',
        n_stops_range=(1, 2)
    )
    if res:
        print_strategy_report(res, c, l)
    tot = time.time() - t0
    print(f"\n{c} total optimization wall time: {tot:.2f}s")

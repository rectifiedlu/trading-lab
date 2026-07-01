# ML Barrier Notes

## Current Target

- Symbol: XAUUSD
- Input: normalized OHLC window from bid candles, plus spread and cyclical time features.
- Label: `1` if `+300 points` hits before `-300 points`; `0` if `-300 points` hits first.
- Ambiguous candles where both barriers hit inside one candle are skipped.
- Current MT5 tick history available through terminal starts around `2026-04-01`, even when requesting more days.

## Important Definitions

- `window`: number of past candles shown to the model.
- `horizon`: number of future candles allowed for either barrier to hit.
- Current evaluator is a labelled-sample evaluator, not a sequential trade simulator. It checks whether model probabilities match precomputed barrier outcomes. It does not yet block overlapping trades or simulate one open position at a time.

## Model Sweep Notes

Latest screening run:

```text
timeframe=30s
barrier=300 points
windows=64,128
horizons=100,200
models=cnn,gru,lstm,transformer,rf,xgb
max_samples=60000
neural epochs=6
tree estimators=300
```

## Rejects So Far

- Linear: too weak, close to random.
- MLP: overfits slightly, confidence does not improve accuracy.
- Transformer: weak in this small setup; mostly low-confidence and negative threshold results.
- CNN in the 6-epoch window/horizon sweep: weaker than the earlier full 12-epoch `w128/h200` run and not stable enough yet.

## Current Contenders

### Random Forest, `window=64`, `horizon=100`, `threshold=0.55`

```text
trades=391
accuracy=64.19%
simplified_pnl=+$244.01
pf=1.48
daily median accuracy=72.97%
bad days exist: May 15 = 27.27%, May 26 = 36.84%, May 27 = 47.73%
```

This is the best simplified PnL from the latest sweep, but daily stability is not clean enough to trust directly.

### XGBoost, `window=64`, `horizon=100`, `threshold=0.65`

```text
trades=1081
accuracy=57.63%
simplified_pnl=+$217.25
pf=1.13
daily median accuracy=61.04%
worst daily accuracy=39.58%
```

More trades and less fragile than RF, but the edge is thinner.

### XGBoost, `window=64`, `horizon=200`, `threshold=0.65`

```text
trades=927
accuracy=57.93%
simplified_pnl=+$205.21
pf=1.15
daily median accuracy=52.50%
worst daily accuracy=43.33%
```

Similar to `h100`, slightly worse median day.

### LSTM, `window=128`, `horizon=100`, `threshold=0.55`

```text
trades=367
accuracy=58.58%
simplified_pnl=+$98.86
pf=1.18
```

Interesting but concentrated in fewer days, so less convincing than RF/XGB from this sweep.

## Best Previous Single Run

Earlier full run:

```text
model=lstm
window=128
horizon=200
epochs=12
threshold=0.65
trades=231
accuracy=71.0%
pf=2.04
simplified_pnl=+$249.02
```

This looked clean, but the later 6-epoch sweep did not reproduce the same high-confidence behavior. Retest with the exact same full settings before treating it as real.

## Next Required Step

Build a sequential ML simulator:

- load saved model
- walk candles in order
- if `P(up) >= threshold`, open long
- if `P(up) <= 1 - threshold`, open short
- hold until +300 or -300 barrier hits on tick/candle path
- no overlapping positions
- track real equity, drawdown, trade list, daily PnL

Do not paper trade from the labelled-sample evaluator alone.

## Sequential Simulator First Pass

Added `forex_ml_signal_sim.py`, which consumes prediction CSVs and tests stateful trading:

- `mode=level`: enter whenever probability is past threshold while flat.
- `mode=change`: enter only when signal changes into long/short.
- `exit=barrier`: exit only on TP/SL barrier.
- `exit=flip`: exit on opposite signal or barrier.
- `exit=flat`: exit when signal disappears or barrier.
- optional cooldown after exit.

Initial results:

### RF `w64/h100`

Command target:

```text
pred=data\forex\forex_ml_barrier_rf_w64_h100_predictions.csv
tp=300
sl=300
```

Best sequential result:

```text
mode=level
threshold=0.55
exit=barrier/flip, same result
cooldown=3
trades=107
win_rate=58.9%
pf=1.19
realised=+$30.30
max_dd=$25.45
median_day=+$6.03
```

This survives, but the edge is much smaller than the labelled-sample result.

### XGB `w64/h100`

Best sequential result:

```text
mode=level
threshold=0.65
exit=flip
cooldown=5
trades=203
win_rate=55.7%
pf=1.04
realised=+$12.14
max_dd=$32.75
median_day=+$1.71
```

This is too thin for now.

## Current ML Conclusion

Labelled-sample metrics are useful for screening, but they overstate edge because they score overlapping possible entries. The first real stateful result says RF `w64/h100` is the only ML candidate worth deeper testing right now, and even that is modest.

## Trade-Outcome Target

Added `--target trade` to `forex_ml_barrier_cnn.py`.

This trains side-specific models:

- `--trade-side long`: label is `1` if a long TP hits before long SL.
- `--trade-side short`: label is `1` if a short TP hits before short SL.

For equal TP/SL, the long model is close to the old `up-first` barrier label and short is close to its complement. This is expected.

Initial trade-target sweep:

```text
models=rf,xgb,lstm
window=64,128
horizon=100,200
tp=300
sl=300
```

Best long-side labelled results:

```text
XGB long w64/h200 th=0.65: trades=275, wr=69.8%, pf=1.91, simplified_pnl=+$270
RF  long w64/h100 th=0.55: trades=391, wr=64.2%, pf=1.48, simplified_pnl=+$244
RF  long w64/h200 th=0.58: trades=149, wr=73.2%, pf=2.26, simplified_pnl=+$180
```

Short-side high-threshold results were mostly weak because the models produced complementary probabilities and rarely gave high standalone short confidence.

Combined long/short sequential simulation was added to `forex_ml_signal_sim.py` with `--long-pred` and `--short-pred`.

RF `w64/h100` combined sequential result:

```text
threshold=0.55
mode=level
exit=barrier/flip
cooldown=3
trades=107
win_rate=58.9%
pf=1.19
realised=+$30.30
max_dd=$25.45
median_day=+$6.03
```

XGB `w64/h200` combined sequential result was negative despite strong labelled metrics:

```text
best realised=-$15.64
pf=0.95
```

Conclusion: trade-target labels did not solve the overlap/state problem by themselves. RF `w64/h100` remains the only current ML candidate that survives stateful simulation, and the edge is still modest.

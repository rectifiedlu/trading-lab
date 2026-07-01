//@version=6
strategy("EMA Pair Cross Confirm TP", overlay=true, pyramiding=0, calc_on_every_tick=true)

fastLen = input.int(8, "Fast EMA", minval=1)
slowLen = input.int(63, "Slow EMA", minval=2)
confirmBars = input.int(2, "Confirm Candles", minval=1)
reverseOnFlip = input.bool(false, "Reverse On Flip (confirm=1 only)")

tpPoints = input.float(400, "Take Profit Points", minval=0)
slPoints = input.float(0, "Stop Loss Points (0 = off)", minval=0)
pointSize = input.float(0.01, "Point Size", minval=0.00001)
blockSession = input.session("2340-0020", "No New Entries Session")

fast = ta.ema(close, fastLen)
slow = ta.ema(close, slowLen)
blockNewEntries = not na(time(timeframe.period, blockSession))

var int longCount = 0
var int shortCount = 0

if fast > slow
    longCount += 1
    shortCount := 0
else if fast < slow
    shortCount += 1
    longCount := 0
else
    longCount := 0
    shortCount := 0

longConfirmed = longCount >= confirmBars
shortConfirmed = shortCount >= confirmBars

longSignal = longConfirmed and not longConfirmed[1]
shortSignal = shortConfirmed and not shortConfirmed[1]

if not blockNewEntries and strategy.position_size == 0 and longSignal
    strategy.entry("Long", strategy.long)

if not blockNewEntries and strategy.position_size == 0 and shortSignal
    strategy.entry("Short", strategy.short)

canReverse = reverseOnFlip and confirmBars == 1

// Opposite EMA regime closes only when explicit SL is disabled.
if slPoints == 0 and strategy.position_size > 0 and fast < slow
    if canReverse
        if not blockNewEntries
            strategy.entry("Short", strategy.short, comment="Flip Short")
        else
            strategy.close("Long", comment="EMA Exit")
    else
        strategy.close("Long", comment="EMA Exit")

if slPoints == 0 and strategy.position_size < 0 and fast > slow
    if canReverse
        if not blockNewEntries
            strategy.entry("Long", strategy.long, comment="Flip Long")
        else
            strategy.close("Short", comment="EMA Exit")
    else
        strategy.close("Short", comment="EMA Exit")

tpDist = tpPoints * pointSize
slDist = slPoints * pointSize

if tpPoints > 0 and strategy.position_size > 0
    strategy.exit("Long TP/SL", "Long", limit=strategy.position_avg_price + tpDist, stop=slPoints > 0 ? strategy.position_avg_price - slDist : na)

if tpPoints > 0 and strategy.position_size < 0
    strategy.exit("Short TP/SL", "Short", limit=strategy.position_avg_price - tpDist, stop=slPoints > 0 ? strategy.position_avg_price + slDist : na)

plot(fast, "Fast EMA", color=color.aqua, linewidth=2)
plot(slow, "Slow EMA", color=color.orange, linewidth=2)

bgcolor(fast > slow ? color.new(color.green, 90) : fast < slow ? color.new(color.red, 90) : na)

plotshape(longSignal, title="Long Confirm", style=shape.triangleup, color=color.lime, size=size.small, location=location.belowbar, text="L")
plotshape(shortSignal, title="Short Confirm", style=shape.triangledown, color=color.red, size=size.small, location=location.abovebar, text="S")

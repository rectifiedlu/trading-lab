// TradingView "Volty Expan Close Strategy" translated for MT5 Strategy Tester.
//
// Pine source:
//   atrs = ta.sma(ta.tr, length) * numATRs
//   strategy.entry("VltClsLE", strategy.long,  stop=close + atrs)
//   strategy.entry("VltClsSE", strategy.short, stop=close - atrs)
//
// Run in MT5 Strategy Tester with "Every tick based on real ticks".
#property strict

#include <Trade/Trade.mqh>

input int    Length      = 5;
input double ATRMult     = 0.75;
input double Lots        = 0.01;
input int    Magic       = 26051602;
input int    DeviationPt = 30;

CTrade trade;
datetime lastBarTime = 0;
double buyStopLevel = 0.0;
double sellStopLevel = 0.0;

bool IsOurPosition()
{
   if(!PositionSelect(_Symbol))
      return false;
   return (int)PositionGetInteger(POSITION_MAGIC) == Magic;
}

int PositionSide()
{
   if(!IsOurPosition())
      return 0;
   long type = PositionGetInteger(POSITION_TYPE);
   if(type == POSITION_TYPE_BUY)
      return 1;
   if(type == POSITION_TYPE_SELL)
      return -1;
   return 0;
}

bool CloseCurrent()
{
   if(!IsOurPosition())
      return true;
   return trade.PositionClose(_Symbol);
}

bool OpenLong()
{
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   return trade.Buy(Lots, _Symbol, ask, 0.0, 0.0, "VltClsLE");
}

bool OpenShort()
{
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   return trade.Sell(Lots, _Symbol, bid, 0.0, 0.0, "VltClsSE");
}

bool ReverseToLong()
{
   if(!CloseCurrent())
      return false;
   return OpenLong();
}

bool ReverseToShort()
{
   if(!CloseCurrent())
      return false;
   return OpenShort();
}

double TrueRange(const int shift)
{
   double high = iHigh(_Symbol, _Period, shift);
   double low = iLow(_Symbol, _Period, shift);
   double prevClose = iClose(_Symbol, _Period, shift + 1);
   if(high == 0.0 || low == 0.0 || prevClose == 0.0)
      return 0.0;
   double a = high - low;
   double b = MathAbs(high - prevClose);
   double c = MathAbs(low - prevClose);
   return MathMax(a, MathMax(b, c));
}

double SMA_TR()
{
   double sum = 0.0;
   for(int i = 1; i <= Length; ++i)
   {
      double tr = TrueRange(i);
      if(tr <= 0.0)
         return 0.0;
      sum += tr;
   }
   return sum / Length;
}

int OnInit()
{
   if(Length < 1 || ATRMult <= 0.0 || Lots <= 0.0)
      return INIT_PARAMETERS_INCORRECT;
   trade.SetExpertMagicNumber(Magic);
   trade.SetDeviationInPoints(DeviationPt);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   buyStopLevel = 0.0;
   sellStopLevel = 0.0;
}

void OnTick()
{
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   int side = PositionSide();

   // Virtual stops mirror Pine stop entries while preventing hedged/multiple
   // positions. Flat: either side can enter. Long: sell stop reverses. Short:
   // buy stop reverses.
   if(buyStopLevel > 0.0 && ask >= buyStopLevel && side <= 0)
   {
      if(side < 0)
         ReverseToLong();
      else
         OpenLong();
      buyStopLevel = 0.0;
      sellStopLevel = 0.0;
      return;
   }

   if(sellStopLevel > 0.0 && bid <= sellStopLevel && side >= 0)
   {
      if(side > 0)
         ReverseToShort();
      else
         OpenShort();
      buyStopLevel = 0.0;
      sellStopLevel = 0.0;
      return;
   }

   datetime barTime = iTime(_Symbol, _Period, 0);
   if(barTime == 0 || barTime == lastBarTime)
      return;
   lastBarTime = barTime;

   // Pine waits until close[length] exists.
   if(Bars(_Symbol, _Period) <= Length + 1)
      return;

   double close1 = iClose(_Symbol, _Period, 1);
   double atrs = SMA_TR() * ATRMult;
   if(close1 <= 0.0 || atrs <= 0.0)
      return;

   double buyStop = NormalizeDouble(close1 + atrs, _Digits);
   double sellStop = NormalizeDouble(close1 - atrs, _Digits);
   side = PositionSide();
   buyStopLevel = side <= 0 ? buyStop : 0.0;
   sellStopLevel = side >= 0 ? sellStop : 0.0;
}

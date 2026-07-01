//+------------------------------------------------------------------+
//|                                       001-Turnaround-Tuesday.mq5 |
//|                        Copyright 2026, Sergei Ermolov (mql5.com) |
//|                        https://www.mql5.com/en/users/dj_ermoloff |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Sergei Ermolov | IT Trader"
#property link      "https://www.mql5.com/en/users/dj_ermoloff"
#property version   "1.1"

#include <Trade\Trade.mqh>
CTrade trade;

enum ENUM_LOT_TYPE {
   LOT_FIXED = 0, //Fixed Lot
   LOT_RISK  = 1  //Risk from Start Balance (%)
};

enum ENUM_ED {
   disabled = 0, //Disabled
   enabled  = 1  //Enabled
};

input ENUM_LOT_TYPE  LotType         = LOT_RISK;  //Lot Type
input double         Lots            = 1;        //Lot Size
input double         Risk            = 2.0;        //Risk per Transaction (%)
input group "=== ATR Filter"
input ENUM_ED        FilterATR       = disabled;   //ATR Filter (Main Bar Size)
input int            ATRPeriod       = 14;         //ATR Period
input double         MinBarATR       = 1.5;        //Min Main Bar Size (x ATR)
input group "=== Inside Bar Filter"
input double         MinBodyPct      = 80.0;       //Min Main Bar Body (% of Range)
input double         MaxInsidePct    = 50.0;       //Max Inside Bar Size (% of Main)
input group "=== Position Settings"
input double         kSLMainBar      = 0.62;       //StopLoss (x Main Bar Range)
input double         rrTP            = 2.5;        //TakeProfit (Risk/Reward)
input group "=== Order Management"
input int            OrderExpiryBars = 3;          //Cancel Pending Order After N Bars
int                  MagicNumber     = 1;


//--- globals
double      balance;
MqlDateTime mtc;
datetime    tc, lastBarTime = 0;
int         atrHandle;
double      point;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
   balance = AccountInfoDouble(ACCOUNT_BALANCE);

   if (iBars(_Symbol, PERIOD_CURRENT) < ATRPeriod + 5) {
      Print("OnInit: not enough history for ", _Symbol,
            " — load at least ", ATRPeriod + 5, " bars and restart");
      return INIT_FAILED;
   }

   atrHandle = iATR(_Symbol, PERIOD_CURRENT, ATRPeriod);
   if (atrHandle == INVALID_HANDLE) {
      Print("OnInit: failed to create ATR handle for ", _Symbol);
      return INIT_FAILED;
   }

   point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   IndicatorRelease(atrHandle);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
   tc = TimeCurrent(mtc);
   
   //Dumb check
   if (AccountInfoDouble(ACCOUNT_EQUITY) < 10) CancelOrders();

   // Run only on new bar
   datetime currentBarTime = (datetime)(tc - tc % PeriodSeconds(PERIOD_CURRENT));
   if (currentBarTime == lastBarTime) return;
   lastBarTime = currentBarTime;

   // Skip if pending order already exists
   if (HasPendingOrder()) return;
   if (PositionsTotal() > 0) return;

   // --- Bar data ---
   // bar[1] = main bar (yesterday / previous closed bar)
   // bar[2] = bar before main (for context if needed)
   // bar[0] = inside bar (just closed bar, current = bar forming)
   // We check: bar[1] is inside bar[2]?  No — classic IB:
   //   bar[2] = main bar, bar[1] = inside bar
   double highMain  = iHigh (_Symbol, PERIOD_CURRENT, 2);
   double lowMain   = iLow  (_Symbol, PERIOD_CURRENT, 2);
   double openMain  = iOpen (_Symbol, PERIOD_CURRENT, 2);
   double closeMain = iClose(_Symbol, PERIOD_CURRENT, 2);

   double highInside  = iHigh (_Symbol, PERIOD_CURRENT, 1);
   double lowInside   = iLow  (_Symbol, PERIOD_CURRENT, 1);

   double mainRange = highMain - lowMain;
   if (mainRange <= 0) return;

   // --- Inside Bar pattern check ---
   // Inside bar must be fully inside main bar (high < main high, low > main low)
   if (highInside >= highMain || lowInside <= lowMain) {
      return; // not an inside bar
   }

   // --- Main bar body filter: body >= MinBodyPct% of range ---
   double bodySize = MathAbs(closeMain - openMain);
   double bodyPct  = bodySize / mainRange * 100.0;
   if (bodyPct < MinBodyPct) {
      Print("OnTick: main bar body too small (", DoubleToString(bodyPct, 1),
            "% < ", MinBodyPct, "%), skip");
      return;
   }

   // --- ATR filter: main bar size >= MinBarATR * ATR ---
   if (FilterATR == enabled) {
      double atrBuf[];
      ArraySetAsSeries(atrBuf, true);
      if (CopyBuffer(atrHandle, 0, 2, 1, atrBuf) <= 0) {
         Print("OnTick: failed to get ATR");
         return;
      }
      double atr      = atrBuf[0];
      if (mainRange < MinBarATR * atr) {
         Print("OnTick: ATR filter — main bar too small (", mainRange / point,
               " pts vs min ", MinBarATR * atr / point, " pts), skip");
         return;
      }
   }

   // --- Inside bar position filter: inside bar range <= MaxInsidePct% of main bar ---
   double insideRange = highInside - lowInside;
   double insidePct = insideRange / mainRange * 100.0;
   if (insidePct > MaxInsidePct) {
      Print("OnTick: inside bar too large (", DoubleToString(insidePct, 1),
            "% > ", MaxInsidePct, "%), skip");
      return;
   }

   // --- SL distance ---
   double slDistance = mainRange * kSLMainBar;
   int    stopsLevel = (int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   double minStop    = MathMax(stopsLevel, 2) * _Point;
   if (slDistance < minStop) slDistance = minStop;
   double slPoints = slDistance / point;

   // --- Lot calculation ---
   double lots;
   if (LotType == LOT_FIXED)
      lots = NormalizeLot(Lots);
   else
      lots = CalcLot(Risk, slPoints);

   if (lots <= 0) {
      Print("OnTick: lot size is 0, skipping");
      return;
   }

   // --- Determine direction based on main bar bias ---
   bool mainBullish = closeMain > openMain;

   // DIR_BOTH: follow main bar direction
   // DIR_BULL: only buy setups
   // DIR_BEAR: only sell setups
   bool doB = mainBullish;
   bool doS = !mainBullish;

   if (doB) {
      double entryBuy = highMain;
      double slBuy    = entryBuy - slDistance;
      double tpBuy    = rrTP > 0 ? entryBuy + slDistance * rrTP : 0;
      double ask      = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double minDist  = stopsLevel * point;
      if(entryBuy < ask + minDist) {
         Print("BuyStop skipped: entry too close to Ask");
         return;
      }
      PlaceBuyStop(lots, entryBuy, slBuy, tpBuy);
   } else if (doS) {
      double entrySell = lowMain;
      double slSell    = entrySell + slDistance;
      double tpSell    = rrTP > 0 ? entrySell - slDistance * rrTP : 0;
      double bid       = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double minDist   = stopsLevel * point;
      if(entrySell > bid - minDist) {
         Print("SellStop skipped: entry too close to Bid");
         return;
      }
      PlaceSellStop(lots, entrySell, slSell, tpSell);
   }
}

//+------------------------------------------------------------------+
//| Place BuyStop order                                              |
//+------------------------------------------------------------------+
bool PlaceBuyStop(double lots, double price, double sl, double tp) {
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(10);

   int    digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   price = NormalizeDouble(price, digits);
   sl    = NormalizeDouble(sl, digits);
   tp    = tp > 0 ? NormalizeDouble(tp, digits) : 0;

   // Expiry = current bar open + OrderExpiryBars * bar period seconds
   datetime barOpen  = (datetime)(TimeCurrent() - TimeCurrent() % PeriodSeconds(PERIOD_CURRENT));
   datetime expiry   = barOpen + (datetime)(OrderExpiryBars * PeriodSeconds(PERIOD_CURRENT));

   bool result = trade.BuyStop(lots, price, _Symbol, sl, tp, ORDER_TIME_SPECIFIED, expiry, "InsideBar");
   if (!result)
      Print("PlaceBuyStop: failed — ", trade.ResultRetcodeDescription());
   else
      Print("PlaceBuyStop: placed @ ", price, " SL=", sl, " TP=", tp, " expiry=", TimeToString(expiry));

   return result;
}

//+------------------------------------------------------------------+
//| Place SellStop order                                             |
//+------------------------------------------------------------------+
bool PlaceSellStop(double lots, double price, double sl, double tp) {
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(10);

   int    digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   price = NormalizeDouble(price, digits);
   sl    = NormalizeDouble(sl, digits);
   tp    = tp > 0 ? NormalizeDouble(tp, digits) : 0;

   datetime barOpen = (datetime)(TimeCurrent() - TimeCurrent() % PeriodSeconds(PERIOD_CURRENT));
   datetime expiry  = barOpen + (datetime)(OrderExpiryBars * PeriodSeconds(PERIOD_CURRENT));

   bool result = trade.SellStop(lots, price, _Symbol, sl, tp, ORDER_TIME_SPECIFIED, expiry, "InsideBar");
   if (!result)
      Print("PlaceSellStop: failed — ", trade.ResultRetcodeDescription());
   else
      Print("PlaceSellStop: placed @ ", price, " SL=", sl, " TP=", tp, " expiry=", TimeToString(expiry));

   return result;
}

//+------------------------------------------------------------------+
//| Check if there is already a pending order for this symbol/magic  |
//+------------------------------------------------------------------+
bool HasPendingOrder() {
   for (int i = 0; i < OrdersTotal(); i++) {
      ulong ticket = OrderGetTicket(i);
      if (ticket == 0) continue;

      ENUM_ORDER_TYPE ot = (ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE);
      if (ot == ORDER_TYPE_BUY_STOP || ot == ORDER_TYPE_SELL_STOP)
         return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Calculate lot size based on risk percentage                      |
//+------------------------------------------------------------------+
double CalcLot(double riskPct, double slPoints) {
   double tickSize  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double lotStep   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double minLot    = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot    = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);

   if(tickSize <= 0 || tickValue <= 0 || slPoints <= 0) {
      Print("CalcLot: invalid symbol specs or SL = 0");
      return 0;
   }

   double riskAmount = balance * riskPct / 100.0;
   double pointValue = (tickValue / tickSize) * point;
   double lots = riskAmount / (slPoints * pointValue);
   lots = MathFloor(lots / lotStep) * lotStep;
   lots = MathMax(minLot, MathMin(maxLot, lots));
   lots = NormalizeDouble(lots, 2);

   // --- Margin check ---
   double margin = 0;
   if(!OrderCalcMargin(ORDER_TYPE_BUY, _Symbol, lots, SymbolInfoDouble(_Symbol, SYMBOL_ASK), margin)) {
      Print("CalcLot: failed to calculate margin");
      return 0;
   }
   double freeMargin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   if(margin > freeMargin) {
      Print("CalcLot: not enough margin — need ", margin, " free ", freeMargin);
      return 0;
   }

   return NormalizeLot(lots);
}

double NormalizeLot(double lots) {
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double minLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   lots = MathFloor(lots / lotStep) * lotStep;
   lots = MathMax(minLot, MathMin(maxLot, lots));
   return NormalizeDouble(lots, 2);
}

void CancelOrders() {
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   int    stopsLevel = (int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   double minDist = MathMax(stopsLevel, 2) * point;

   for(int i = OrdersTotal() - 1; i >= 0; i--) {
      ulong ticket = OrderGetTicket(i);
      if(ticket == 0) continue;

      ENUM_ORDER_TYPE ot = (ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE);
      double orderPrice = OrderGetDouble(ORDER_PRICE_OPEN);

      if(ot == ORDER_TYPE_BUY_STOP  && orderPrice <= ask + minDist) continue;
      if(ot == ORDER_TYPE_SELL_STOP && orderPrice >= bid - minDist) continue;

      if(!trade.OrderDelete(ticket))
         Print("CancelOrders: failed to delete #", ticket, " — ", trade.ResultRetcodeDescription());
   }
}
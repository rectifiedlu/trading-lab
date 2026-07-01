//+------------------------------------------------------------------+
//|                                     Universal Breakout Study.mq5 |
//|                        Copyright 2026, Sergei Ermolov (mql5.com) |
//|                        https://www.mql5.com/en/users/dj_ermoloff |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Sergei Ermolov | IT Trader"
#property link      "https://www.mql5.com/en/users/dj_ermoloff"
#property version   "1.1"

#include <Trade\Trade.mqh>
#include <CBoxSession.mqh>


//+------------------------------------------------------------------+
//| Enums                                                            |
//+------------------------------------------------------------------+
enum ENUM_ED
  {
   disabled = 0, //Disabled
   enabled = 1, //Enabled
  };

enum ENUM_GMT
  {
   gmt0 = -11, //GMT -11
   gmt1 = -10, //GMT -10
   gmt2 = -9,  //GMT -9
   gmt3 = -8,  //GMT -8
   gmt4 = -7,  //GMT -7
   gmt5 = -6,  //GMT -6
   gmt6 = -5,  //GMT -5
   gmt7 = -4,  //GMT -4
   gmt8 = -3,  //GMT -3
   gmt9 = -2,  //GMT -2
   gmt10 = -1, //GMT -1
   gmt11 = 0,  //GMT 0
   gmt12 = 1,  //GMT +1
   gmt13 = 2,  //GMT +2
   gmt14 = 3,  //GMT +3
   gmt15 = 4,  //GMT +4
   gmt16 = 5,  //GMT +5
   gmt17 = 6,  //GMT +6
   gmt18 = 7,  //GMT +7
   gmt19 = 8,  //GMT +8
   gmt20 = 9,  //GMT +9
   gmt21 = 10, //GMT +10
   gmt22 = 11, //GMT +11
  };
enum ENUM_STOP
  {
   sl0 = 0, //Null
   sl1 = 1, //Fixed
   sl2 = 2, //Coefficient from box
  };
enum ENUM_TP
  {
   tp0 = 0, //Null
   tp1 = 1, //Fixed
   tp2 = 2, //Coefficient from box
  };

//+------------------------------------------------------------------+
//| Inputs                                                           |
//+------------------------------------------------------------------+
input group "=== Box settings ===";
input ENUM_GMT    GMT             = gmt14;       //GMT offset
input int         StartHourBox    = 0;            //Box start hour (GMT)
input int         TotalBarBox     = 48;           //Box size (candles)

input group "=== Open Settings ===";
input int         Shift           = 0;            //Deviation from extremes (points)
input int         Expiration_Minute = 1110;       //Order expiration time (0 = off)
input ENUM_ED     CancelOpposite = enabled;       //Cancel opposite order on entry

input group "=== StopLoss Settings ===";
input ENUM_STOP   StopLoss_Type   = sl2;          //StopLoss type
input double      StopLoss        = 62;           //Fixed (points)
input double      k_StopLoss      = 0.61;         //Coefficient from box

input group "=== TakeProfit Settings ===";
input ENUM_TP     TakeProfit_Type = tp2;          //TakeProfit type
input double      TakeProfit      = 30;           //Fixed TakeProfit (points)
input double      k_TakeProfit    = 1.0;          //Coefficient from box

input group "=== StopLoss Management ===";
input group "--- Breakeven ---";
input ENUM_ED     Paritet_Type = 0;                //Use Break-Even
input int         Paritet = 22;                    //Break-Even distance
input int         Pips = 10;                       //Break-Even profit

input group "--- Classical Trailling Stop ---";
input ENUM_ED     OnTraillingStop = 1;             //Use Trailling Stop
input int         TraillingStop = 12;              //Trailling distance
input int         TraillingStep = 1;               //Trailling step
input int         TraillingStart = 10;             //Trailling min. profit

input group "=== Time Exit ===";
input ENUM_ED     Close_Type = enabled;            //Use Time Exit
input int         Market_Minute = 70;              //Check After (minutes)
input int         Close_Level = 0;                 //Minimum Profit to Exit (points)

input group "=== Days trading ===";
input ENUM_ED     Monday = false;
input ENUM_ED     Tuesday = true;
input ENUM_ED     Wednesday = true;
input ENUM_ED     Thursday = true;
input ENUM_ED     Friday = true;

input group "=== Trade Settings ===";
input double      RiskPercent     = 1.0;          //Risk per trade (%)
int   slippage    = 0;            //Slippage (points)
int   MagicNumber = 32463651;     //Magic number
string _comment   = "UB";         //Trade comment

//+------------------------------------------------------------------+
//| Globals                                                          |
//+------------------------------------------------------------------+
CTrade       trade;
CBoxSession* session;
double       balance;
datetime     tc;
MqlDateTime  mtc;
double       Ask, Bid;

datetime     last_box_time = 0; // end_time last box
double _point;
int    _digits;

//+------------------------------------------------------------------+
//| OnInit                                                           |
//+------------------------------------------------------------------+
int OnInit()
  {
   if(_Period != PERIOD_H1) Print("Attach the EA to an H1 chart"); 
   
   if(iBars(_Symbol, PERIOD_H1) < TotalBarBox + 5)
     {
      Print("OnInit: not enough H1 history for ", _Symbol);
      return INIT_FAILED;
     }  

   balance = AccountInfoDouble(ACCOUNT_BALANCE);
   _digits  = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   _point   = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   if(_digits == 5 || _digits == 3)
      _point *= 10;

   session = new CBoxSession(
      StartHourBox,
      0,
      TotalBarBox,
      PERIOD_H1,
      Expiration_Minute,
      GMT
   );

   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
//| OnDeinit                                                         |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(session != NULL)
     {
      delete session;
      session = NULL;
     }
   EventKillTimer();
  }

//+------------------------------------------------------------------+
//| OnTick                                                           |
//+------------------------------------------------------------------+
void OnTick()
  {
   Ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   Bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   tc  = TimeCurrent(mtc);

   LoopPositions();

   session.Tick(Ask, Bid);

   if(!session.IsReady())
      return;

   if(session.box.end_time == last_box_time)
      return;

   if(session.box.upper_cancel && session.box.lower_cancel)
     {
      last_box_time = session.box.end_time;
      return;
     }

   if(!IsTradingDay(session.box.end_time))
     {
      last_box_time = session.box.end_time;
      return;
     }

   PlaceOrders();
   last_box_time = session.box.end_time;
  }

//+------------------------------------------------------------------+
//| Place buy stop and sell stop on new box                          |
//+------------------------------------------------------------------+
void PlaceOrders()
  {
   SB_Box box     = session.box;
   double boxSize = box.high - box.low;
   double stopsDist = (int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _point;

   datetime expiration = (Expiration_Minute > 0)
                         ? box.end_time + Expiration_Minute * 60
                         : 0;

   // --- Buy Stop ---
   if(!box.upper_cancel)
     {
      double entry = NormalizeDouble(box.high + Shift * _point, _digits);
      if(entry < Ask + stopsDist)
         Print("Buy Stop skipped: entry ", entry, " too close to Ask ", Ask, " (stops level: ", stopsDist, ")");
      else
        {
         double sl    = CalcSL(entry, ORDER_TYPE_BUY_STOP, boxSize);
         double tp    = CalcTP(entry, ORDER_TYPE_BUY_STOP, boxSize);
         double slPts = (sl > 0) ? (entry - sl) / _point : 0;
         double lots  = CalcLot(RiskPercent, slPts);
         if(!CheckMargin(lots, ORDER_TYPE_BUY_STOP, entry)) return;
         OpenBuyStop(lots, entry, sl, tp, MagicNumber, _comment, expiration);
        }
     }

   // --- Sell Stop ---
   if(!box.lower_cancel)
     {
      double entry = NormalizeDouble(box.low - Shift * _point, _digits);
      if(entry > Bid - stopsDist)
         Print("Sell Stop skipped: entry ", entry, " too close to Bid ", Bid, " (stops level: ", stopsDist, ")");
      else
        {
         double sl    = CalcSL(entry, ORDER_TYPE_SELL_STOP, boxSize);
         double tp    = CalcTP(entry, ORDER_TYPE_SELL_STOP, boxSize);
         double slPts = (sl > 0) ? (sl - entry) / _point : 0;
         double lots  = CalcLot(RiskPercent, slPts);
         if(!CheckMargin(lots, ORDER_TYPE_SELL_STOP, entry)) return;
         OpenSellStop(lots, entry, sl, tp, MagicNumber, _comment, expiration);
        }
     }
  }

//+------------------------------------------------------------------+
//| Calculate Stop Loss price                                        |
//+------------------------------------------------------------------+
double CalcSL(double entry, ENUM_ORDER_TYPE type, double boxSize)
  {
   if(StopLoss_Type == sl0)
      return 0;
   double dist = 0;

   if(StopLoss_Type == sl1)
      dist = StopLoss * _point;
   else
      if(StopLoss_Type == sl2)
         dist = boxSize * k_StopLoss;

   if(dist <= 0)
      return 0;

   double sl = (type == ORDER_TYPE_BUY_STOP)
               ? entry - dist
               : entry + dist;

   return NormalizeDouble(sl, _digits);
  }

//+------------------------------------------------------------------+
//| Calculate Take Profit price                                      |
//+------------------------------------------------------------------+
double CalcTP(double entry, ENUM_ORDER_TYPE type, double boxSize)
  {
   if(TakeProfit_Type == tp0)
      return 0;
   double dist   = 0;

   if(TakeProfit_Type == tp1)
      dist = TakeProfit * _point;
   else
      if(TakeProfit_Type == tp2)
         dist = boxSize * k_TakeProfit;

   if(dist <= 0)
      return 0;

   double tp = (type == ORDER_TYPE_BUY_STOP)
               ? entry + dist
               : entry - dist;

   return NormalizeDouble(tp, _digits);
  }

//+------------------------------------------------------------------+
//| Calculate lot size                                               |
//+------------------------------------------------------------------+
double CalcLot(double riskPct, double slPoints)
  {
   double riskAmount = balance * riskPct / 100.0;
   double tickSize   = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double tickValue  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double lotStep    = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double minLot     = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot     = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);

   if(tickSize <= 0 || tickValue <= 0 || slPoints <= 0)
     {
      Print("CalcLot: invalid symbol specs or SL = 0");
      return minLot;
     }

   double pointValue = (tickValue / tickSize) * SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double lots = riskAmount / (slPoints * pointValue);
   lots = MathFloor(lots / lotStep) * lotStep;
   lots = MathMax(minLot, MathMin(maxLot, lots));
   return NormalizeDouble(lots, 2);
  }

//+------------------------------------------------------------------+
//| Loop through all open positions                                  |
//+------------------------------------------------------------------+
void LoopPositions()
  {
   int total = PositionsTotal();
   for(int i = total - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;

      long     type       = PositionGetInteger(POSITION_TYPE);
      double   lots       = PositionGetDouble(POSITION_VOLUME);
      double   open_price = PositionGetDouble(POSITION_PRICE_OPEN);
      double   sl         = PositionGetDouble(POSITION_SL);
      double   tp         = PositionGetDouble(POSITION_TP);
      datetime open_time  = (datetime)PositionGetInteger(POSITION_TIME);
      double   cur_price  = (type == POSITION_TYPE_BUY) ? Bid : Ask;
      double   profit_pts = (type == POSITION_TYPE_BUY)
                            ? (cur_price - open_price) / _point
                            : (open_price - cur_price) / _point;
                            
      if(CancelOpposite == enabled) CancelOppositeOrder((ENUM_ORDER_TYPE)type);

      // --- Breakeven ---
      if(Paritet_Type == enabled)
        {
         if(profit_pts >= Paritet)
           {
            double be_sl = (type == POSITION_TYPE_BUY)
                           ? NormalizeDouble(open_price + Pips * _point, _digits)
                           : NormalizeDouble(open_price - Pips * _point, _digits);
                           
            double stopsMin = (int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _point;
            bool valid_sl = (type == POSITION_TYPE_BUY)
                            ? (cur_price - be_sl >= stopsMin)
                            : (be_sl - cur_price >= stopsMin);

            bool already_be = (type == POSITION_TYPE_BUY)
                              ? (sl >= be_sl)
                              : (sl <= be_sl && sl > 0);

            if(!already_be && valid_sl)
              {
               if(!trade.PositionModify(ticket, be_sl, tp))
                  Print("Breakeven: failed to modify #", ticket, " — ", trade.ResultRetcodeDescription());
              }
           }
        }

      // --- Trailing Stop ---
      if(OnTraillingStop == enabled)
        {
         if(profit_pts >= TraillingStop + TraillingStart)
           {
            double new_sl = (type == POSITION_TYPE_BUY)
                            ? NormalizeDouble(cur_price - TraillingStop * _point, _digits)
                            : NormalizeDouble(cur_price + TraillingStop * _point, _digits);
            double stopsMin = (int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _point;
            bool valid_sl = (type == POSITION_TYPE_BUY)
                            ? (cur_price - new_sl >= stopsMin)
                            : (new_sl - cur_price >= stopsMin);
            
            if(valid_sl)
               {         
               bool need_modify = false;
               if(type == POSITION_TYPE_BUY)
                 {
                  need_modify = (sl == 0 || new_sl >= sl + TraillingStep * _point);
                 }
               else
                 {
                  need_modify = (sl == 0 || new_sl <= sl - TraillingStep * _point);
                 }
   
               if(need_modify)
                 {
                  if(!trade.PositionModify(ticket, new_sl, tp))
                     Print("Trailing: failed to modify #", ticket, " — ", trade.ResultRetcodeDescription());
                 }
              }
           }
        }

      // --- Close by time + profit ---
      if(Close_Type == enabled)
        {
         if(tc - open_time >= Market_Minute * 60)
           {
            if(profit_pts >= Close_Level)
              {
               if(!trade.PositionClose(ticket))
                  Print("LoopPositions: failed to close #", ticket, " — ", trade.ResultRetcodeDescription());
               continue;
              }
           }
        }
     }
  }

//+------------------------------------------------------------------+
//| Open a Buy Stop order                                            |
//+------------------------------------------------------------------+
bool OpenBuyStop(double lots, double price, double sl = 0, double tp = 0, ulong magic = 0, string comment = "", datetime expiration = 0)
  {
   trade.SetExpertMagicNumber(magic);
   trade.SetDeviationInPoints(slippage);

   price = NormalizeDouble(price, _digits);
   if(sl > 0)
      sl = NormalizeDouble(sl, _digits);
   if(tp > 0)
      tp = NormalizeDouble(tp, _digits);

   bool result = trade.BuyStop(lots, price, _Symbol,
                               sl > 0 ? sl : 0,
                               tp > 0 ? tp : 0,
                               expiration > 0 ? ORDER_TIME_SPECIFIED : ORDER_TIME_GTC,
                               expiration, comment);

   if(!result)
      Print("OpenBuyStop: failed — ", trade.ResultRetcodeDescription());

   return result;
  }

//+------------------------------------------------------------------+
//| Open a Sell Stop order                                           |
//+------------------------------------------------------------------+
bool OpenSellStop(double lots, double price, double sl = 0, double tp = 0, ulong magic = 0, string comment = "", datetime expiration = 0)
  {
   trade.SetExpertMagicNumber(magic);
   trade.SetDeviationInPoints(slippage);

   price = NormalizeDouble(price, _digits);
   if(sl > 0)
      sl = NormalizeDouble(sl, _digits);
   if(tp > 0)
      tp = NormalizeDouble(tp, _digits);

   bool result = trade.SellStop(lots, price, _Symbol,
                                sl > 0 ? sl : 0,
                                tp > 0 ? tp : 0,
                                expiration > 0 ? ORDER_TIME_SPECIFIED : ORDER_TIME_GTC,
                                expiration, comment);

   if(!result)
      Print("OpenSellStop: failed — ", trade.ResultRetcodeDescription());

   return result;
  }

//+------------------------------------------------------------------+
//| Check if current box day is allowed for trading                  |
//+------------------------------------------------------------------+
bool IsTradingDay(datetime box_start)
  {
   MqlDateTime dt;
   TimeToStruct(box_start, dt);
   switch(dt.day_of_week)
     {
      case 1:
         return (bool)Monday;
      case 2:
         return (bool)Tuesday;
      case 3:
         return (bool)Wednesday;
      case 4:
         return (bool)Thursday;
      case 5:
         return (bool)Friday;
      default:
         return false;
     }
  }
  
bool CheckMargin(double lots, ENUM_ORDER_TYPE type, double price)
  {
   if (AccountInfoDouble(ACCOUNT_EQUITY) < 10) return false;
   double margin = 0;
   if(!OrderCalcMargin(type, _Symbol, lots, price, margin))
     {
      Print("CheckMargin: failed to calculate margin");
      return false;
     }
   if(margin > AccountInfoDouble(ACCOUNT_MARGIN_FREE))
     {
      Print("CheckMargin: not enough margin — required=", margin, " free=", AccountInfoDouble(ACCOUNT_MARGIN_FREE));
      return false;
     }
   return true;
  }
  
void CancelOppositeOrder(ENUM_ORDER_TYPE triggered_type)
  {
   ENUM_ORDER_TYPE opposite = (triggered_type == ORDER_TYPE_BUY)
                              ? ORDER_TYPE_SELL_STOP
                              : ORDER_TYPE_BUY_STOP;

   for(int i = OrdersTotal() - 1; i >= 0; i--)
     {
      ulong ticket = OrderGetTicket(i);
      if(ticket == 0) continue;
      if(OrderGetString(ORDER_SYMBOL) != _Symbol) continue;
      if(OrderGetInteger(ORDER_MAGIC) != MagicNumber) continue;
      if(OrderGetInteger(ORDER_TYPE) != opposite) continue;

      if(!trade.OrderDelete(ticket))
         Print("CancelOppositeOrder: failed to delete #", ticket, " — ", trade.ResultRetcodeDescription());
     }
  }

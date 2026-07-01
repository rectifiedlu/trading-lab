//+------------------------------------------------------------------+
//| XANDER Gold Recovery                                             |
//| Version: 1.1          | Date: 2026-04-23                        |
//| Platform: MT5                                                    |
//| Symbol:   XAUUSD                                                 |
//+------------------------------------------------------------------+
// CHANGELOG:
// v1.0 - 2026-04-23 - Initial release
// v1.1 - 2026-04-23 - Added CheckMoneyForTrade before every order
//+------------------------------------------------------------------+
// Community channel link included in the file header.
//+------------------------------------------------------------------+
#property copyright "XANDER Systems"
#property link      "https://t.me/xandertool"
#property version   "1.10"
#property description "XANDER Gold Recovery - Keltner Channel strategy for XAUUSD."
#property description "Includes optional Progressive Recovery System with basket controls."
#property description "Source code provided for educational and research purposes."
//+------------------------------------------------------------------+
//| Includes                                                         |
//+------------------------------------------------------------------+
#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Trade\AccountInfo.mqh>
//+------------------------------------------------------------------+
//| Inputs - Lot Size                                                |
//+------------------------------------------------------------------+
input double xr_FixedLot = 0.01;              // Base lot size (step 0.01)
//+------------------------------------------------------------------+
//| Inputs - Progressive Recovery                                    |
//+------------------------------------------------------------------+
input bool   xr_UseRecovery        = true;    // Enable Progressive Recovery
input double xr_RecoveryDistance   = 300;     // Distance between orders (pips)
input double xr_RecoveryMultiplier = 1.4;     // Lot multiplier per step
input int    xr_RecoveryMaxOrders  = 5;       // Max orders in basket (safety)
//+------------------------------------------------------------------+
//| Inputs - Basket Protection                                       |
//+------------------------------------------------------------------+
input double xr_BasketTakeProfit = 30.0;      // Basket TP in account currency (0=off)
input double xr_BasketHardStop   = 200.0;     // Basket Hard Stop in currency (0=off)
//+------------------------------------------------------------------+
//| Inputs - Trade Exit                                              |
//+------------------------------------------------------------------+
input double xr_StopLoss   = 0;               // Stop Loss in pips (0 = disabled)
input double xr_TakeProfit = 500;             // Take Profit in pips (0 = disabled)
//+------------------------------------------------------------------+
//| Inputs - Trailing Stop                                           |
//+------------------------------------------------------------------+
input double xr_TrailTrigger = 300;           // Trailing trigger (pips, 0 = off)
input double xr_TrailStop    = 300;           // Trailing stop (pips)
input double xr_TrailStep    = 100;           // Trailing step (pips)
//+------------------------------------------------------------------+
//| Inputs - Session Filter                                          |
//+------------------------------------------------------------------+
input bool xr_UseTimeFilter = true;           // Enable session filter
input int  xr_StartHour     = 2;              // Session start hour
input int  xr_StartMinute   = 30;             // Session start minute
input int  xr_EndHour       = 21;             // Session end hour
input int  xr_EndMinute     = 0;              // Session end minute
//+------------------------------------------------------------------+
//| Inputs - Entry Signal                                            |
//+------------------------------------------------------------------+
input int xr_KeltnerPeriod = 50;              // Keltner Channel length
input int xr_EmaFast       = 10;              // Fast EMA period
input int xr_EmaSlow       = 200;             // Slow EMA period
//+------------------------------------------------------------------+
//| Inputs - Execution                                               |
//+------------------------------------------------------------------+
input int    xr_MaxSlippage = 3;              // Max slippage (pips)
input double xr_MaxSpread   = 65;             // Max spread in points (0 = unlimited)
//+------------------------------------------------------------------+
//| Inputs - General                                                 |
//+------------------------------------------------------------------+
input int    xr_Magic   = 260423;             // Magic number
input string xr_Comment = "XANDER_GR";        // Trade comment
//+------------------------------------------------------------------+
//| Trade objects                                                    |
//+------------------------------------------------------------------+
CTrade          m_trade;
CPositionInfo   m_position;
CSymbolInfo     m_symbol;
CAccountInfo    m_account;
//+------------------------------------------------------------------+
//| Indicator handles and buffers                                    |
//+------------------------------------------------------------------+
int    h_emaFast = INVALID_HANDLE;
int    h_emaSlow = INVALID_HANDLE;
int    h_keltner = INVALID_HANDLE;
double buf_emaFast[];
double buf_emaSlow[];
double buf_keltner[];
//+------------------------------------------------------------------+
//| Runtime state                                                    |
//+------------------------------------------------------------------+
double   g_pip2double;
int      g_pip2points;
int      g_slippage;
datetime g_lastBarTime;
//+------------------------------------------------------------------+
//| OnInit                                                           |
//+------------------------------------------------------------------+
int OnInit()
  {
   if(_Digits % 2 == 1)
     {
      g_pip2double = _Point * 10;
      g_pip2points = 10;
      g_slippage   = 10 * xr_MaxSlippage;
     }
   else
     {
      g_pip2double = _Point;
      g_pip2points = 1;
      g_slippage   = xr_MaxSlippage;
     }

   if(!m_symbol.Name(_Symbol))
     {
      Print("XANDER Gold Recovery: Failed to initialize symbol.");
      return INIT_FAILED;
     }
   m_symbol.RefreshRates();

   if(xr_UseRecovery)
     {
      if(xr_RecoveryMaxOrders < 1 || xr_RecoveryMaxOrders > 20)
        {
         Print("XANDER Gold Recovery: RecoveryMaxOrders must be between 1 and 20.");
         return INIT_PARAMETERS_INCORRECT;
        }
      if(xr_RecoveryMultiplier < 1.0)
        {
         Print("XANDER Gold Recovery: RecoveryMultiplier must be 1.0 or greater.");
         return INIT_PARAMETERS_INCORRECT;
        }
      if(xr_RecoveryDistance <= 0)
        {
         Print("XANDER Gold Recovery: RecoveryDistance must be greater than 0.");
         return INIT_PARAMETERS_INCORRECT;
        }
     }

   m_trade.SetExpertMagicNumber(xr_Magic);
   m_trade.SetDeviationInPoints(g_slippage);
   m_trade.SetTypeFillingBySymbol(_Symbol);
   m_trade.LogLevel(LOG_LEVEL_ERRORS);

   h_emaFast = iMA(_Symbol, _Period, xr_EmaFast, 0, MODE_EMA, PRICE_CLOSE);
   h_emaSlow = iMA(_Symbol, _Period, xr_EmaSlow, 0, MODE_EMA, PRICE_CLOSE);
   h_keltner = iMA(_Symbol, _Period, xr_KeltnerPeriod, 0, MODE_EMA, PRICE_CLOSE);

   if(h_emaFast == INVALID_HANDLE || h_emaSlow == INVALID_HANDLE || h_keltner == INVALID_HANDLE)
     {
      Print("XANDER Gold Recovery: Failed to create indicator handles.");
      return INIT_FAILED;
     }

   ArraySetAsSeries(buf_emaFast, true);
   ArraySetAsSeries(buf_emaSlow, true);
   ArraySetAsSeries(buf_keltner, true);

   g_lastBarTime = 0;
   Print("XANDER Gold Recovery v1.0 initialized.");
   return INIT_SUCCEEDED;
  }
//+------------------------------------------------------------------+
//| OnDeinit                                                         |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(h_emaFast != INVALID_HANDLE) IndicatorRelease(h_emaFast);
   if(h_emaSlow != INVALID_HANDLE) IndicatorRelease(h_emaSlow);
   if(h_keltner != INVALID_HANDLE) IndicatorRelease(h_keltner);
  }
//+------------------------------------------------------------------+
//| OnTick                                                           |
//+------------------------------------------------------------------+
void OnTick()
  {
   if(!m_symbol.RefreshRates()) return;
   if(xr_MaxSpread > 0 && m_symbol.Spread() > xr_MaxSpread) return;

   ManageBasket();
   ManageTrailingStop();

   if(xr_UseRecovery && BasketCount() > 0)
     {
      CheckRecoveryEntry();
      return;
     }

   datetime bar_time = (datetime)SeriesInfoInteger(_Symbol, _Period, SERIES_LASTBAR_DATE);
   if(bar_time == g_lastBarTime) return;
   g_lastBarTime = bar_time;

   if(xr_UseTimeFilter && !IsInTradingWindow()) return;
   if(BasketCount() > 0) return;

   if(!LoadIndicatorBuffers()) return;
   int signal = GetSignal();
   if(signal == 1)       OpenFirstBuy();
   else if(signal == -1) OpenFirstSell();
  }
//+------------------------------------------------------------------+
//| Load indicator buffers                                           |
//+------------------------------------------------------------------+
bool LoadIndicatorBuffers()
  {
   if(CopyBuffer(h_emaFast, 0, 0, 3, buf_emaFast) <= 0) return false;
   if(CopyBuffer(h_emaSlow, 0, 0, 3, buf_emaSlow) <= 0) return false;
   if(CopyBuffer(h_keltner, 0, 0, 3, buf_keltner) <= 0) return false;
   return true;
  }
//+------------------------------------------------------------------+
//| Channel range                                                    |
//+------------------------------------------------------------------+
double GetChannelRange()
  {
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   int bars = xr_KeltnerPeriod;
   if(CopyRates(_Symbol, _Period, 0, bars, rates) <= 0) return 0.0;
   double sum = 0.0;
   for(int i = 0; i < bars; i++) sum += (rates[i].high - rates[i].low);
   return sum / bars;
  }
//+------------------------------------------------------------------+
//| Signal detection                                                 |
//+------------------------------------------------------------------+
int GetSignal()
  {
   double range = GetChannelRange();
   if(range <= 0) return 0;

   double mid     = buf_keltner[1];
   double upper   = mid + range;
   double lower   = mid - range;
   double close1  = iClose(_Symbol, _Period, 1);
   double close2  = iClose(_Symbol, _Period, 2);
   double emaFast = buf_emaFast[1];
   double emaSlow = buf_emaSlow[1];

   bool buySetup  = (close2 < lower) && (close1 > lower) && (emaFast > emaSlow);
   bool sellSetup = (close2 > upper) && (close1 < upper) && (emaFast < emaSlow);

   if(buySetup)  return 1;
   if(sellSetup) return -1;
   return 0;
  }
//+------------------------------------------------------------------+
//| First buy entry                                                  |
//+------------------------------------------------------------------+
void OpenFirstBuy()
  {
   double ask = m_symbol.Ask();
   double sl  = (xr_StopLoss   > 0) ? ask - xr_StopLoss   * g_pip2double : 0.0;
   double tp  = (xr_TakeProfit > 0) ? ask + xr_TakeProfit * g_pip2double : 0.0;
   double volume = NormalizeVolume(xr_FixedLot);

   if(!CheckMoneyForTrade(ORDER_TYPE_BUY, volume, ask)) return;

   if(!m_trade.Buy(volume, _Symbol, ask, sl, tp, xr_Comment))
      Print("XANDER Gold Recovery: Buy failed - ", m_trade.ResultRetcodeDescription());
  }
//+------------------------------------------------------------------+
//| First sell entry                                                 |
//+------------------------------------------------------------------+
void OpenFirstSell()
  {
   double bid = m_symbol.Bid();
   double sl  = (xr_StopLoss   > 0) ? bid + xr_StopLoss   * g_pip2double : 0.0;
   double tp  = (xr_TakeProfit > 0) ? bid - xr_TakeProfit * g_pip2double : 0.0;
   double volume = NormalizeVolume(xr_FixedLot);

   if(!CheckMoneyForTrade(ORDER_TYPE_SELL, volume, bid)) return;

   if(!m_trade.Sell(volume, _Symbol, bid, sl, tp, xr_Comment))
      Print("XANDER Gold Recovery: Sell failed - ", m_trade.ResultRetcodeDescription());
  }
//+------------------------------------------------------------------+
//| Recovery entry check                                             |
//+------------------------------------------------------------------+
void CheckRecoveryEntry()
  {
   int count = BasketCount();
   if(count <= 0) return;
   if(count >= xr_RecoveryMaxOrders) return;

   ulong  last_ticket = 0;
   double last_price  = 0.0;
   double last_volume = 0.0;
   ENUM_POSITION_TYPE last_type = POSITION_TYPE_BUY;
   datetime last_time = 0;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(!m_position.SelectByIndex(i)) continue;
      if(m_position.Symbol() != _Symbol || m_position.Magic() != xr_Magic) continue;
      if(m_position.Time() >= last_time)
        {
         last_time   = m_position.Time();
         last_ticket = m_position.Ticket();
         last_price  = m_position.PriceOpen();
         last_volume = m_position.Volume();
         last_type   = (ENUM_POSITION_TYPE)m_position.PositionType();
        }
     }

   if(last_ticket == 0) return;

   double distance = xr_RecoveryDistance * g_pip2double;
   double ask = m_symbol.Ask();
   double bid = m_symbol.Bid();
   double next_volume = NormalizeVolume(last_volume * xr_RecoveryMultiplier);

   if(last_type == POSITION_TYPE_BUY)
     {
      if(ask <= last_price - distance)
        {
         if(!CheckMoneyForTrade(ORDER_TYPE_BUY, next_volume, ask)) return;
         if(!m_trade.Buy(next_volume, _Symbol, ask, 0.0, 0.0, xr_Comment))
            Print("XANDER Gold Recovery: Recovery Buy failed - ", m_trade.ResultRetcodeDescription());
        }
     }
   else if(last_type == POSITION_TYPE_SELL)
     {
      if(bid >= last_price + distance)
        {
         if(!CheckMoneyForTrade(ORDER_TYPE_SELL, next_volume, bid)) return;
         if(!m_trade.Sell(next_volume, _Symbol, bid, 0.0, 0.0, xr_Comment))
            Print("XANDER Gold Recovery: Recovery Sell failed - ", m_trade.ResultRetcodeDescription());
        }
     }
  }
//+------------------------------------------------------------------+
//| Manage basket                                                    |
//+------------------------------------------------------------------+
void ManageBasket()
  {
   if(BasketCount() == 0) return;

   double basket_profit = BasketProfit();

   if(xr_BasketTakeProfit > 0 && basket_profit >= xr_BasketTakeProfit)
     {
      CloseBasket("Basket TP hit");
      return;
     }

   if(xr_BasketHardStop > 0 && basket_profit <= -xr_BasketHardStop)
     {
      CloseBasket("Basket Hard Stop hit");
      return;
     }
  }
//+------------------------------------------------------------------+
//| Close all basket positions                                       |
//+------------------------------------------------------------------+
void CloseBasket(string reason)
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(!m_position.SelectByIndex(i)) continue;
      if(m_position.Symbol() != _Symbol || m_position.Magic() != xr_Magic) continue;
      if(!m_trade.PositionClose(m_position.Ticket()))
         Print("XANDER Gold Recovery: Close failed - ", m_trade.ResultRetcodeDescription());
     }
   Print("XANDER Gold Recovery: ", reason);
  }
//+------------------------------------------------------------------+
//| Count basket positions                                           |
//+------------------------------------------------------------------+
int BasketCount()
  {
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
      if(m_position.SelectByIndex(i))
         if(m_position.Symbol() == _Symbol && m_position.Magic() == xr_Magic)
            count++;
   return count;
  }
//+------------------------------------------------------------------+
//| Sum floating profit of basket                                    |
//+------------------------------------------------------------------+
double BasketProfit()
  {
   double total = 0.0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(!m_position.SelectByIndex(i)) continue;
      if(m_position.Symbol() != _Symbol || m_position.Magic() != xr_Magic) continue;
      total += m_position.Profit() + m_position.Swap() + m_position.Commission();
     }
   return total;
  }
//+------------------------------------------------------------------+
//| Check if there is enough free margin for a trade                 |
//+------------------------------------------------------------------+
bool CheckMoneyForTrade(ENUM_ORDER_TYPE type, double volume, double price)
  {
   double free_margin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   double required_margin = 0.0;

   if(!OrderCalcMargin(type, _Symbol, volume, price, required_margin))
      return false;

   if(required_margin > free_margin)
      return false;

   return true;
  }
//+------------------------------------------------------------------+
//| Normalize volume to symbol step                                  |
//+------------------------------------------------------------------+
double NormalizeVolume(double lot)
  {
   double min_lot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_lot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double step_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(step_lot <= 0.0) step_lot = 0.01;

   lot = MathRound(lot / step_lot) * step_lot;
   if(lot < min_lot) lot = min_lot;
   if(lot > max_lot) lot = max_lot;
   return NormalizeDouble(lot, 2);
  }
//+------------------------------------------------------------------+
//| Trading session window                                           |
//+------------------------------------------------------------------+
bool IsInTradingWindow()
  {
   MqlDateTime now;
   TimeToStruct(TimeCurrent(), now);
   int cur_min   = now.hour * 60 + now.min;
   int start_min = xr_StartHour * 60 + xr_StartMinute;
   int end_min   = xr_EndHour   * 60 + xr_EndMinute;
   if(start_min < end_min) return (cur_min >= start_min && cur_min < end_min);
   return (cur_min >= start_min || cur_min < end_min);
  }
//+------------------------------------------------------------------+
//| Trailing stop                                                    |
//+------------------------------------------------------------------+
void ManageTrailingStop()
  {
   if(xr_TrailTrigger <= 0) return;
   if(xr_UseRecovery && BasketCount() > 1) return;

   double trigger = xr_TrailTrigger * g_pip2double;
   double stop    = xr_TrailStop    * g_pip2double;
   double step    = xr_TrailStep    * g_pip2double;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(!m_position.SelectByIndex(i)) continue;
      if(m_position.Symbol() != _Symbol || m_position.Magic() != xr_Magic) continue;

      double open_price = m_position.PriceOpen();
      double cur_sl     = m_position.StopLoss();
      double cur_tp     = m_position.TakeProfit();
      double bid        = m_symbol.Bid();
      double ask        = m_symbol.Ask();

      if(m_position.PositionType() == POSITION_TYPE_BUY)
        {
         double profit = bid - open_price;
         if(profit < trigger) continue;
         double new_sl = bid - stop;
         if(cur_sl == 0.0 || new_sl - cur_sl >= step)
            m_trade.PositionModify(m_position.Ticket(), new_sl, cur_tp);
        }
      else if(m_position.PositionType() == POSITION_TYPE_SELL)
        {
         double profit = open_price - ask;
         if(profit < trigger) continue;
         double new_sl = ask + stop;
         if(cur_sl == 0.0 || cur_sl - new_sl >= step)
            m_trade.PositionModify(m_position.Ticket(), new_sl, cur_tp);
        }
     }
  }
//+------------------------------------------------------------------+

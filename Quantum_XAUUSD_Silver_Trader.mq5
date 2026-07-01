//+------------------------------------------------------------------+
//| Quantum XAUUSD Silver Trader                                     |
//| Version: 1.0          | Date: 2026-06-02                        |
//| Platform: MT5                                                    |
//| Developer: Cristhian Gaibor                                     |
//+------------------------------------------------------------------+
// CHANGELOG:
// v1.0 - 2026-06-02 - Initial release. Multi-indicator EA for Gold
//        and Silver with adaptive ATR trailing stop and capital
//        protection. Per-symbol presets for XAUUSD and XAGUSD.
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Cristhian Gaibor"
#property link      "https://www.mql5.com/en/users/09151993a"
#property version   "1.00"
#property description "Multi-indicator EA for Gold (XAUUSD) and Silver (XAGUSD)."
#property description "Adaptive ATR trailing stop and built-in capital protection."

#include <Trade\Trade.mqh>
#include <Trade\AccountInfo.mqh>

//==================================================================
// INPUTS
// Edit order, top to bottom, follows what a user normally changes:
//   1) Money management      what you risk
//   2) Entry signal          how it enters
//   3) Stop loss / Take prof what closes a trade
//   4) Trailing stop         how profit is protected
//   5) Capital protection    safety nets
//   6) Symbol presets        auto-tuned values for Gold / Silver
//   7) Adaptive engine        advanced, leave default
//   8) Optimization           strategy tester only
//==================================================================

//------------------------------------------------------------------
// 1) MONEY MANAGEMENT
//------------------------------------------------------------------
input string sep_money = "══════ 1) MONEY MANAGEMENT ══════"; // ─────────────
input double InpBaseRisk            = 0.5;   // Base risk per trade (%)
input double InpMaxRisk             = 2.0;   // Maximum risk per trade (%)
input bool   InpUsePositionSizeLimit= true;  // Limit position size by equity
input double InpMaxPositionPct      = 5.0;   // Max position size (% of equity)
input double InpRiskIncreaseFactor  = 1.2;   // Risk x after a win
input double InpRiskDecreaseFactor  = 0.7;   // Risk x after a loss
input int    InpMaxConsecutiveLosses= 2;     // Max consecutive losses before cut
input bool   InpMicroMode           = true;  // Micro account mode (small deposits)
input double InpMicroCorrection     = 0.6;   // Micro account risk correction
input bool   InpIgnoreMarginMinLot  = true;  // Allow min lot even if over margin limit

//------------------------------------------------------------------
// 2) ENTRY SIGNAL
//------------------------------------------------------------------
input string sep_signal = "══════ 2) ENTRY SIGNAL ══════"; // ─────────────────
input ENUM_TIMEFRAMES InpTimeframe  = PERIOD_M15; // Analysis timeframe
input double InpSignalThreshold     = 0.65;  // Signal decision threshold
input int    InpLearningPeriod      = 200;   // Adaptive learning window
input int    InpRSIPeriod           = 14;    // RSI period
input bool   InpUseRSI              = true;  // Use RSI module
input double InpRSIWeight           = 2.0;   // RSI weight
input int    InpADXPeriod           = 14;    // ADX period
input bool   InpUseADX              = true;  // Use ADX module
input double InpADXWeight           = 1.0;   // ADX weight
input int    InpFastMAPeriod        = 50;    // Fast MA period
input int    InpSlowMAPeriod        = 200;   // Slow MA period
input bool   InpUseMA               = true;  // Use MA module
input double InpMAWeight            = 1.0;   // MA weight
input int    InpChaosPeriod         = 20;    // Volatility window
input bool   InpUseChaosFilter      = true;  // Block entries in low volatility
input double InpChaosThreshold      = 0.35;  // Default volatility threshold
input bool   InpUseChaosInSignal    = true;  // Use volatility in signal
input double InpChaosWeight         = 1.0;   // Volatility weight
input bool   InpUseState            = true;  // Use adaptive state module
input double InpStateWeight         = 1.0;   // Adaptive state weight

//------------------------------------------------------------------
// 3) STOP LOSS & TAKE PROFIT
//------------------------------------------------------------------
input string sep_sltp = "══════ 3) STOP LOSS & TAKE PROFIT ══════"; // ─────────
input int    InpATRPeriod           = 14;    // ATR period (SL/TP/trailing base)
input double InpATRMultSL           = 4.0;   // Default ATR multiplier for SL
input double InpTPtoSL              = 0.7;   // Default ATR multiplier for TP
input int    InpMinBarsBetweenTrades= 3;     // Default min bars between trades

//------------------------------------------------------------------
// 4) TRAILING STOP
//------------------------------------------------------------------
input string sep_trail = "══════ 4) TRAILING STOP ══════"; // ─────────────────
input bool   InpUseTrailing         = true;  // Enable adaptive trailing stop
input double InpBaseTrailATRMult    = 1.5;   // Base trailing ATR multiplier
input double InpMaxTrailATRMult     = 4.0;   // Max trailing ATR multiplier
input double InpTrailActivation     = 1.0;   // Activation profit (in ATR)
input double InpChaosSensitivity    = 0.7;   // Volatility sensitivity
input double InpStateInfluence      = 0.5;   // Adaptive state influence

//------------------------------------------------------------------
// 5) CAPITAL PROTECTION
//------------------------------------------------------------------
input string sep_prot = "══════ 5) CAPITAL PROTECTION ══════"; // ─────────────
input bool   InpUseEquityProtection = true;  // Enable equity protection
input double InpMaxDailyDD           = 10.0;  // Max daily drawdown (%)
input double InpMaxTotalDD           = 50.0;  // Max total drawdown (%)
input double InpDDBuffer             = 2.0;   // Drawdown warning buffer (%)
input bool   InpUseDailyLossLimit    = true;  // Enable daily loss limit
input double InpDailyLossPct         = 15.0;  // Daily loss limit (%)
input double InpDailyLossAbs         = 50.0;  // Daily loss limit (absolute)
input double InpDailyLossATRMult     = 3.0;   // ATR multiplier for loss limit
input bool   InpUseHardStop          = false; // Enable hard stop (close all)
input double InpHardStopLevel        = 20.0;  // Hard stop drawdown level (%)

//------------------------------------------------------------------
// 6) SYMBOL PRESETS
// When enabled, these override risk / SL / TP / trailing automatically
// if the chart symbol is Gold or Silver. Disable to use the values above
// on every symbol.
//------------------------------------------------------------------
input string sep_preset = "══════ 6) SYMBOL PRESETS ══════"; // ───────────────
input bool   InpUseSymbolPresets    = true;  // Auto-apply Gold/Silver presets

input string sep_gold = "──── Gold (XAUUSD) ────"; // ──────────────────────────
input double InpGoldATRMultSL       = 8.0;   // Gold ATR multiplier for SL
input double InpGoldTPtoSL          = 0.7;   // Gold ATR multiplier for TP
input double InpGoldBaseRisk        = 0.5;   // Gold base risk (%)
input int    InpGoldMinBars         = 1;     // Gold min bars between trades
input double InpGoldTrailATRMult    = 1.5;   // Gold base trailing ATR multiplier
input double InpGoldChaosThreshold  = 0.35;  // Gold volatility threshold

input string sep_silver = "──── Silver (XAGUSD) ────"; // ─────────────────────
input double InpSilverATRMultSL     = 2.0;   // Silver ATR multiplier for SL
input double InpSilverTPtoSL        = 2.5;   // Silver ATR multiplier for TP
input double InpSilverBaseRisk      = 0.3;   // Silver base risk (%)
input int    InpSilverMinBars       = 2;     // Silver min bars between trades
input double InpSilverTrailATRMult  = 2.0;   // Silver base trailing ATR multiplier
input double InpSilverChaosThreshold= 0.4;   // Silver volatility threshold

//------------------------------------------------------------------
// 7) ADAPTIVE STATE ENGINE (ADVANCED)
// Experimental tuning. Leave at defaults if you are not optimizing.
//------------------------------------------------------------------
input string sep_state = "══════ 7) ADAPTIVE STATE ENGINE ══════"; // ──────────
input int    InpStates              = 45;    // Number of internal states
input double InpStateDecayRate       = 0.03;  // State decay rate
input double InpStateUpdateRate       = 0.1;   // State update rate
input double InpStateMemoryFactor     = 0.9;   // State memory factor

//------------------------------------------------------------------
// 8) OPTIMIZATION (Strategy Tester only)
//------------------------------------------------------------------
input string sep_opt = "══════ 8) OPTIMIZATION ══════"; // ────────────────────
input bool   InpAutoOptimize        = false; // Auto-optimize in Strategy Tester (experimental)
input int    InpOptimizationPasses  = 50;    // Optimization passes

//==================================================================
// INTERNAL WORKING COPIES (overwritten by presets / optimization)
//==================================================================
int      XP_States;
double   XP_StateDecayRate;
double   XP_StateUpdateRate;
double   XP_StateMemoryFactor;
int      XP_RSIPeriod;
int      XP_ADXPeriod;
int      XP_ATRPeriod;
int      XP_FastMAPeriod;
int      XP_SlowMAPeriod;
double   XP_BaseRisk;
double   XP_ATRMultSL;
double   XP_TPtoSL;
int      XP_MinBars;
double   XP_BaseTrailATRMult;
double   XP_ChaosThreshold;

//==================================================================
// GLOBAL STATE
//==================================================================
CTrade        XP_Trade;
CAccountInfo  XP_Account;

int      XP_State = 0;
double   XP_WaveFunction[];
double   XP_StateReturns[];
datetime XP_LastTradeTime;
double   XP_PositionMatrix[][2];
int      XP_LossCounter = 0;
double   XP_CurrentRisk = 0.5;
double   XP_AIWeights[];
double   XP_MarketMatrix[][4];
int      XP_SignalHistory[];

int      XP_RSIHandle, XP_ADXHandle, XP_ATRHandle, XP_MAFastHandle, XP_MASlowHandle;

double   XP_LastRSI = 0, XP_LastADX = 0, XP_LastATR = 0, XP_LastFastMA = 0, XP_LastSlowMA = 0;
datetime XP_LastCacheTime;

double   XP_InitialEquity = 0;
double   XP_MaxEquityToday = 0;
double   XP_MinEquityToday = 0;
datetime XP_LastEquityUpdate = 0;
double   XP_TotalProfitToday = 0;
double   XP_DailyLossLimit;
bool     XP_TradingHalted = false;
int      XP_MarginErrorCount = 0;
datetime XP_LastErrorTime = 0;

datetime XP_LastTrailTime = 0;
double   XP_LastTrailPrice = 0;
double   XP_LastTrailSl = 0;

bool     XP_MicroMode;

bool     XP_IsFirstRun = true;
double   XP_BestParams[][2];
bool     XP_OptimizationCompleted = false;


//==================================================================
// SECTION 1 - GENERIC HELPERS
//==================================================================
//+------------------------------------------------------------------+
//| Human-readable error description                                 |
//+------------------------------------------------------------------+
string XP_ErrorDescription(int error_code)
{
   switch(error_code)
   {
      case 0:    return("No error");
      case 1:    return("No error, unknown result");
      case 2:    return("Common error");
      case 3:    return("Invalid trade parameters");
      case 4:    return("Trade server is busy");
      case 6:    return("No connection to trade server");
      case 7:    return("Not enough rights");
      case 8:    return("Too frequent requests");
      case 64:   return("Account disabled");
      case 65:   return("Invalid account");
      case 128:  return("Trade timeout");
      case 129:  return("Invalid price");
      case 130:  return("Invalid stops");
      case 131:  return("Invalid trade volume");
      case 132:  return("Market is closed");
      case 133:  return("Trade is disabled");
      case 134:  return("Not enough money");
      case 135:  return("Price changed");
      case 136:  return("Off quotes");
      case 138:  return("Requote");
      case 146:  return("Trade context is busy");
      case 147:  return("Expiration denied by broker");
      case 148:  return("Too many open and pending orders");
      case 4756: return("Invalid stops");
      default:   return("Unknown error: " + IntegerToString(error_code));
   }
}

//+------------------------------------------------------------------+
//| Arithmetic mean                                                  |
//+------------------------------------------------------------------+
double XP_Mean(double &array[])
{
   int count = ArraySize(array);
   if(count == 0) return 0;
   double sum = 0.0;
   for(int i = 0; i < count; i++) sum += array[i];
   return sum / count;
}

//+------------------------------------------------------------------+
//| Standard deviation                                               |
//+------------------------------------------------------------------+
double XP_StdDev(double &array[])
{
   int count = ArraySize(array);
   if(count == 0) return 0;
   double mean = XP_Mean(array);
   double sum = 0.0;
   for(int i = 0; i < count; i++) sum += MathPow(array[i] - mean, 2);
   return MathSqrt(sum / count);
}

//+------------------------------------------------------------------+
//| Normalize price to tick size                                     |
//+------------------------------------------------------------------+
double XP_NormalizePrice(double price)
{
   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(tickSize == 0) return price;
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   return NormalizeDouble(MathRound(price / tickSize) * tickSize, digits);
}

//+------------------------------------------------------------------+
//| Read one indicator value by handle                               |
//+------------------------------------------------------------------+
double XP_GetIndicatorValue(int handle, int buffer = 0, int shift = 0)
{
   double value[1];
   if(CopyBuffer(handle, buffer, shift, 1, value) != 1)
   {
      Print("Failed to read indicator data: ", GetLastError());
      return EMPTY_VALUE;
   }
   return value[0];
}

//+------------------------------------------------------------------+
//| Check free margin before opening an order                        |
//+------------------------------------------------------------------+
bool XP_CheckMoneyForTrade(string symb, double lots, ENUM_ORDER_TYPE type)
{
   MqlTick tick;
   if(!SymbolInfoTick(symb, tick)) return false;
   double price = (type == ORDER_TYPE_BUY) ? tick.ask : tick.bid;

   double margin;
   if(!OrderCalcMargin(type, symb, lots, price, margin))
   {
      Print("OrderCalcMargin failed: ", GetLastError());
      return false;
   }

   double free_margin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   if(margin > free_margin)
   {
      PrintFormat("Not enough free margin: need %.2f, have %.2f", margin, free_margin);
      return false;
   }
   return true;
}

//+------------------------------------------------------------------+
//| Is the market currently open for this symbol?                    |
//| Prevents "[Market closed]" failures on trade and modify ops.     |
//+------------------------------------------------------------------+
bool XP_MarketOpen()
{
   if(SymbolInfoInteger(_Symbol, SYMBOL_TRADE_MODE) != SYMBOL_TRADE_MODE_FULL)
      return false;

   MqlDateTime t;
   TimeToStruct(TimeCurrent(), t);
   ENUM_DAY_OF_WEEK day = (ENUM_DAY_OF_WEEK)t.day_of_week;
   int secOfDay = t.hour * 3600 + t.min * 60 + t.sec;

   datetime from, to;
   for(int i = 0; SymbolInfoSessionTrade(_Symbol, day, i, from, to); i++)
   {
      int sFrom = (int)(from % 86400);
      int sTo   = (int)(to   % 86400);
      if(secOfDay >= sFrom && secOfDay <= sTo)
         return true;
   }
   return false;
}


//==================================================================
// SECTION 2 - PARAMETERS & OPTIMIZATION
//==================================================================
//+------------------------------------------------------------------+
//| Auto-correct out-of-range parameters                             |
//+------------------------------------------------------------------+
void XP_AutoCorrectParameters()
{
   if(XP_FastMAPeriod >= XP_SlowMAPeriod)
   {
      int old = XP_SlowMAPeriod;
      XP_SlowMAPeriod = XP_FastMAPeriod + 10;
      PrintFormat("Auto-correct: SlowMAPeriod %d -> %d (must be > FastMAPeriod)", old, XP_SlowMAPeriod);
   }

   if(XP_ATRPeriod < 10)
   {
      int old = XP_ATRPeriod;
      XP_ATRPeriod = 10;
      PrintFormat("Auto-correct: ATRPeriod %d -> %d (minimum 10)", old, XP_ATRPeriod);
   }

   if(XP_States < 5)
      PrintFormat("Warning: States=%d is too low (>=5 recommended)", XP_States);

   if(XP_BaseRisk > InpMaxRisk)
   {
      double old = XP_BaseRisk;
      XP_BaseRisk = InpMaxRisk;
      PrintFormat("Auto-correct: BaseRisk %.2f -> %.2f (cannot exceed MaxRisk)", old, XP_BaseRisk);
   }
}

//+------------------------------------------------------------------+
//| Save optimized parameters to file                                |
//+------------------------------------------------------------------+
void XP_SaveOptimizedParams()
{
   int handle = FileOpen("QGST_Params_" + _Symbol + ".bin", FILE_WRITE|FILE_BIN);
   if(handle == INVALID_HANDLE) return;

   FileWriteInteger(handle, XP_States);
   FileWriteDouble(handle, XP_StateDecayRate);
   FileWriteDouble(handle, XP_StateUpdateRate);
   FileWriteDouble(handle, XP_StateMemoryFactor);
   FileWriteInteger(handle, XP_RSIPeriod);
   FileWriteInteger(handle, XP_ADXPeriod);
   FileWriteInteger(handle, XP_ATRPeriod);
   FileWriteInteger(handle, XP_FastMAPeriod);
   FileWriteInteger(handle, XP_SlowMAPeriod);
   FileWriteDouble(handle, XP_BaseRisk);
   FileWriteDouble(handle, XP_ATRMultSL);
   FileWriteDouble(handle, XP_TPtoSL);

   FileClose(handle);
   Print("Optimized parameters saved");
}

//+------------------------------------------------------------------+
//| Load optimized parameters from file                              |
//+------------------------------------------------------------------+
void XP_LoadOptimizedParams()
{
   int handle = FileOpen("QGST_Params_" + _Symbol + ".bin", FILE_READ|FILE_BIN);
   if(handle == INVALID_HANDLE) return;

   XP_States           = FileReadInteger(handle);
   XP_StateDecayRate   = FileReadDouble(handle);
   XP_StateUpdateRate  = FileReadDouble(handle);
   XP_StateMemoryFactor= FileReadDouble(handle);
   XP_RSIPeriod        = FileReadInteger(handle);
   XP_ADXPeriod        = FileReadInteger(handle);
   XP_ATRPeriod        = FileReadInteger(handle);
   XP_FastMAPeriod     = FileReadInteger(handle);
   XP_SlowMAPeriod     = FileReadInteger(handle);
   XP_BaseRisk         = FileReadDouble(handle);
   XP_ATRMultSL        = FileReadDouble(handle);
   XP_TPtoSL           = FileReadDouble(handle);

   FileClose(handle);
   Print("Optimized parameters loaded");
}

//+------------------------------------------------------------------+
//| Random-search auto-optimization (tester only)                    |
//+------------------------------------------------------------------+
void XP_RunAutoOptimization()
{
   if(!MQLInfoInteger(MQL_TESTER) || !InpAutoOptimize) return;

   Print("Starting auto-optimization...");

   ArrayResize(XP_BestParams, 12);
   double bestResult = -DBL_MAX;

   for(int pass = 0; pass < InpOptimizationPasses; pass++)
   {
      XP_States            = MathRand() % 46 + 5;
      XP_StateDecayRate    = MathRand()/32767.0*0.09 + 0.01;
      XP_StateUpdateRate   = MathRand()/32767.0*0.25 + 0.05;
      XP_StateMemoryFactor = MathRand()/32767.0*0.19 + 0.8;
      XP_RSIPeriod         = MathRand() % 15 + 7;
      XP_ADXPeriod         = MathRand() % 16 + 10;
      XP_ATRPeriod         = MathRand() % 11 + 10;
      XP_FastMAPeriod      = MathRand() % 81 + 20;
      XP_SlowMAPeriod      = MathRand() % 201 + 100;
      XP_BaseRisk          = MathRand()/32767.0*0.9 + 0.1;
      XP_ATRMultSL         = MathRand()/32767.0*2.0 + 1.0;
      XP_TPtoSL            = MathRand()/32767.0*1.5 + 1.5;

      XP_AutoCorrectParameters();

      double result = OnTester();
      if(result > bestResult)
      {
         bestResult = result;
         XP_BestParams[0][1]  = XP_States;
         XP_BestParams[1][1]  = XP_StateDecayRate;
         XP_BestParams[2][1]  = XP_StateUpdateRate;
         XP_BestParams[3][1]  = XP_StateMemoryFactor;
         XP_BestParams[4][1]  = XP_RSIPeriod;
         XP_BestParams[5][1]  = XP_ADXPeriod;
         XP_BestParams[6][1]  = XP_ATRPeriod;
         XP_BestParams[7][1]  = XP_FastMAPeriod;
         XP_BestParams[8][1]  = XP_SlowMAPeriod;
         XP_BestParams[9][1]  = XP_BaseRisk;
         XP_BestParams[10][1] = XP_ATRMultSL;
         XP_BestParams[11][1] = XP_TPtoSL;
      }
   }

   XP_States            = (int)XP_BestParams[0][1];
   XP_StateDecayRate    = XP_BestParams[1][1];
   XP_StateUpdateRate   = XP_BestParams[2][1];
   XP_StateMemoryFactor = XP_BestParams[3][1];
   XP_RSIPeriod         = (int)XP_BestParams[4][1];
   XP_ADXPeriod         = (int)XP_BestParams[5][1];
   XP_ATRPeriod         = (int)XP_BestParams[6][1];
   XP_FastMAPeriod      = (int)XP_BestParams[7][1];
   XP_SlowMAPeriod      = (int)XP_BestParams[8][1];
   XP_BaseRisk          = XP_BestParams[9][1];
   XP_ATRMultSL         = XP_BestParams[10][1];
   XP_TPtoSL            = XP_BestParams[11][1];

   XP_AutoCorrectParameters();
   Print("Auto-optimization complete. Best score: ", bestResult);
   XP_SaveOptimizedParams();
   XP_OptimizationCompleted = true;
}


//==================================================================
// SECTION 3 - MARKET ANALYSIS & SIGNAL
//==================================================================
//+------------------------------------------------------------------+
//| Market volatility measure                                        |
//+------------------------------------------------------------------+
double XP_CalculateChaos()
{
   int bars = MathMin(InpChaosPeriod, Bars(_Symbol, InpTimeframe));
   if(bars < InpChaosPeriod) return 0.5;

   double returns[];
   ArrayResize(returns, InpChaosPeriod);
   for(int i = 0; i < InpChaosPeriod; i++)
      returns[i] = MathLog(iClose(_Symbol, InpTimeframe, i)) - MathLog(iClose(_Symbol, InpTimeframe, i+1));

   double mean = XP_Mean(returns);
   double sd   = XP_StdDev(returns);
   return (sd != 0) ? MathAbs(mean / sd) : 0.5;
}

//+------------------------------------------------------------------+
//| Refresh indicator cache once per bar                             |
//+------------------------------------------------------------------+
void XP_UpdateIndicatorCache()
{
   datetime currentTime = iTime(_Symbol, InpTimeframe, 0);
   if(currentTime != XP_LastCacheTime)
   {
      XP_LastRSI    = XP_GetIndicatorValue(XP_RSIHandle);
      XP_LastADX    = XP_GetIndicatorValue(XP_ADXHandle, 0);
      XP_LastATR    = XP_GetIndicatorValue(XP_ATRHandle);
      XP_LastFastMA = XP_GetIndicatorValue(XP_MAFastHandle);
      XP_LastSlowMA = XP_GetIndicatorValue(XP_MASlowHandle);
      XP_LastCacheTime = currentTime;
   }
}

//+------------------------------------------------------------------+
//| Initialize indicators and adaptive weights                       |
//+------------------------------------------------------------------+
void XP_InitSignalEngine()
{
   ArrayResize(XP_AIWeights, 6);
   ArrayInitialize(XP_AIWeights, 0.5);

   XP_AutoCorrectParameters();

   string fileName = "QGST_Weights_" + _Symbol + ".txt";
   int fileHandle = FileOpen(fileName, FILE_READ|FILE_TXT|FILE_COMMON);
   if(fileHandle != INVALID_HANDLE)
   {
      for(int i = 0; i < 6 && !FileIsEnding(fileHandle); i++)
         XP_AIWeights[i] = StringToDouble(FileReadString(fileHandle));
      FileClose(fileHandle);
      Print("Adaptive weights loaded from file");
   }
   else
   {
      Print("No weights file found, using defaults");
   }

   ArrayResize(XP_MarketMatrix, InpLearningPeriod);
   ArrayResize(XP_SignalHistory, InpLearningPeriod);
   ArrayInitialize(XP_SignalHistory, 0);

   XP_RSIHandle    = iRSI(_Symbol, InpTimeframe, XP_RSIPeriod, PRICE_CLOSE);
   XP_ADXHandle    = iADX(_Symbol, InpTimeframe, XP_ADXPeriod);
   XP_ATRHandle    = iATR(_Symbol, InpTimeframe, XP_ATRPeriod);
   XP_MAFastHandle = iMA(_Symbol, InpTimeframe, XP_FastMAPeriod, 0, MODE_SMA, PRICE_CLOSE);
   XP_MASlowHandle = iMA(_Symbol, InpTimeframe, XP_SlowMAPeriod, 0, MODE_SMA, PRICE_CLOSE);

   if(XP_RSIHandle == INVALID_HANDLE || XP_ADXHandle == INVALID_HANDLE ||
      XP_ATRHandle == INVALID_HANDLE || XP_MAFastHandle == INVALID_HANDLE ||
      XP_MASlowHandle == INVALID_HANDLE)
   {
      Print("Failed to create indicator handles: ", GetLastError());
   }
}

//+------------------------------------------------------------------+
//| Generate a trade signal                                          |
//+------------------------------------------------------------------+
int XP_GenerateTradeSignal()
{
   if(XP_TradingHalted) return -1;

   XP_UpdateIndicatorCache();

   if(XP_LastRSI == EMPTY_VALUE || XP_LastADX == EMPTY_VALUE ||
      XP_LastATR == EMPTY_VALUE || XP_LastFastMA == EMPTY_VALUE ||
      XP_LastSlowMA == EMPTY_VALUE)
   {
      Print("Insufficient indicator data");
      return -1;
   }

   double chaos = XP_CalculateChaos();
   if(InpUseChaosFilter && chaos < XP_ChaosThreshold) return -1;

   XP_State = MathRand() % XP_States;

   double signalStrength = 0;

   if(InpUseRSI)
      signalStrength += InpRSIWeight * XP_AIWeights[0] * (XP_LastRSI - 50.0) / 50.0;

   if(InpUseADX)
      signalStrength += InpADXWeight * XP_AIWeights[1] * (XP_LastADX - 20.0) / 30.0;

   if(InpUseMA)
      signalStrength += InpMAWeight * XP_AIWeights[2] * (XP_LastFastMA - XP_LastSlowMA) / SymbolInfoDouble(_Symbol, SYMBOL_POINT);

   if(InpUseChaosInSignal)
      signalStrength += InpChaosWeight * XP_AIWeights[3] * chaos;

   if(InpUseState)
      signalStrength += InpStateWeight * XP_AIWeights[4] * (XP_State - XP_States/2.0) / XP_States;

   if(signalStrength >  InpSignalThreshold) return ORDER_TYPE_BUY;
   if(signalStrength < -InpSignalThreshold) return ORDER_TYPE_SELL;

   return -1;
}

//+------------------------------------------------------------------+
//| Adaptive learning update after each trade                        |
//+------------------------------------------------------------------+
void XP_UpdateLearning(int signal, double profit)
{
   for(int i = InpLearningPeriod-1; i > 0; i--)
      XP_SignalHistory[i] = XP_SignalHistory[i-1];
   XP_SignalHistory[0] = (profit > 0) ? signal : -signal;

   double rsi   = XP_GetIndicatorValue(XP_RSIHandle) / 100.0;
   double adx   = XP_GetIndicatorValue(XP_ADXHandle, 0) / 100.0;
   double atr   = XP_GetIndicatorValue(XP_ATRHandle) / 100.0;
   double chaos = XP_CalculateChaos();

   for(int i = InpLearningPeriod-1; i > 0; i--)
   {
      XP_MarketMatrix[i][0] = XP_MarketMatrix[i-1][0];
      XP_MarketMatrix[i][1] = XP_MarketMatrix[i-1][1];
      XP_MarketMatrix[i][2] = XP_MarketMatrix[i-1][2];
      XP_MarketMatrix[i][3] = XP_MarketMatrix[i-1][3];
   }
   XP_MarketMatrix[0][0] = rsi;
   XP_MarketMatrix[0][1] = adx;
   XP_MarketMatrix[0][2] = atr;
   XP_MarketMatrix[0][3] = chaos;

   double successRate = 0;
   int count = 0;
   for(int i = 0; i < MathMin(100, InpLearningPeriod); i++)
   {
      if(XP_SignalHistory[i] != 0)
      {
         count++;
         if(XP_SignalHistory[i] > 0) successRate++;
      }
   }
   successRate = (count > 0) ? successRate / count : 0.5;

   double adjustment = (successRate - 0.5) * 0.1;
   for(int i = 0; i < 6; i++)
      XP_AIWeights[i] = MathMin(1.0, MathMax(0.1, XP_AIWeights[i] + adjustment));
}


//==================================================================
// SECTION 4 - RISK & MONEY MANAGEMENT
//==================================================================
//+------------------------------------------------------------------+
//| Adaptive risk based on volatility                                |
//+------------------------------------------------------------------+
double XP_CalculateAdaptiveRisk()
{
   double chaos = XP_CalculateChaos();
   double risk  = XP_BaseRisk * (1.0 + MathSin(chaos * M_PI));

   if(XP_MicroMode)
   {
      double equity = XP_Account.Equity();
      if(equity < 1000)
         risk *= InpMicroCorrection * MathSqrt(equity/1000);
   }

   return MathMin(InpMaxRisk, MathMax(0.1, risk));
}

//+------------------------------------------------------------------+
//| Lot size with margin awareness                                   |
//+------------------------------------------------------------------+
double XP_CalculateLotSize(double riskPercent, double price, double slDistance)
{
   double balance    = XP_Account.Balance();
   double riskAmount = balance * (riskPercent / 100.0);
   double tickValue  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double volumeStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double minLot     = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);

   if(slDistance == 0 || tickValue == 0) return minLot;

   double lotSize = riskAmount / (slDistance / SymbolInfoDouble(_Symbol, SYMBOL_POINT) * tickValue);
   lotSize = MathRound(lotSize / volumeStep) * volumeStep;
   lotSize = MathMin(lotSize, SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX));
   lotSize = MathMax(lotSize, minLot);
   lotSize = NormalizeDouble(lotSize, 2);

   double marginRequired;
   if(OrderCalcMargin(ORDER_TYPE_BUY, _Symbol, lotSize, price, marginRequired))
   {
      double freeMargin = XP_Account.FreeMargin();
      while(marginRequired > freeMargin && lotSize > minLot)
      {
         lotSize = NormalizeDouble(lotSize - volumeStep, 2);
         if(!OrderCalcMargin(ORDER_TYPE_BUY, _Symbol, lotSize, price, marginRequired)) break;
      }
      if(marginRequired > freeMargin)
      {
         if(XP_MicroMode && InpIgnoreMarginMinLot)
         {
            if(OrderCalcMargin(ORDER_TYPE_BUY, _Symbol, minLot, price, marginRequired) && marginRequired <= freeMargin)
               return minLot;
         }
         PrintFormat("Not enough margin for min lot %.2f. Need %.2f %s, free %.2f %s",
                     minLot, marginRequired, XP_Account.Currency(), freeMargin, XP_Account.Currency());
         return -1;
      }
   }
   return lotSize;
}

//+------------------------------------------------------------------+
//| Limit position size by equity                                    |
//+------------------------------------------------------------------+
bool XP_CheckPositionSize(double lotSize, double price)
{
   if(!InpUsePositionSizeLimit) return true;

   double margin_required;
   if(!OrderCalcMargin(ORDER_TYPE_BUY, _Symbol, lotSize, price, margin_required))
   {
      Print("Margin calc failed: ", GetLastError());
      return true;
   }

   double max_allowed = XP_Account.Equity() * (InpMaxPositionPct / 100.0);

   if(margin_required > max_allowed)
   {
      if(InpIgnoreMarginMinLot &&
         lotSize <= SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN))
      {
         PrintFormat("Minimum lot (%.2f) allowed through position-size cap.", lotSize);
         return true;
      }
      PrintFormat("Position cost exceeded: %.2f %s > %.2f %s",
                  margin_required, XP_Account.Currency(), max_allowed, XP_Account.Currency());
      return false;
   }
   return true;
}


//==================================================================
// SECTION 5 - CAPITAL PROTECTION
//==================================================================
//+------------------------------------------------------------------+
//| Track daily equity high/low and profit                           |
//+------------------------------------------------------------------+
void XP_UpdateDailyEquity()
{
   datetime current = TimeCurrent();
   MqlDateTime now;
   TimeToStruct(current, now);

   static MqlDateTime lastDay;
   static bool firstRun = true;

   if(firstRun || (now.day != lastDay.day || now.mon != lastDay.mon || now.year != lastDay.year))
   {
      firstRun = false;
      lastDay  = now;
      XP_InitialEquity   = XP_Account.Equity();
      XP_MaxEquityToday  = XP_InitialEquity;
      XP_MinEquityToday  = XP_InitialEquity;
      XP_TotalProfitToday= 0;
      XP_LastEquityUpdate= current;

      double atr = XP_GetIndicatorValue(XP_ATRHandle);
      double atrBasedLimit = (atr != EMPTY_VALUE) ? atr * InpDailyLossATRMult : InpDailyLossAbs;

      XP_DailyLossLimit = MathMin(
         XP_Account.Balance() * (InpDailyLossPct / 100.0),
         MathMax(InpDailyLossAbs, atrBasedLimit));

      XP_TradingHalted = false;
      return;
   }

   double equity = XP_Account.Equity();
   if(equity > XP_MaxEquityToday || XP_MaxEquityToday == 0) XP_MaxEquityToday = equity;
   if(equity < XP_MinEquityToday || XP_MinEquityToday == 0) XP_MinEquityToday = equity;

   if(current - XP_LastEquityUpdate >= 60)
   {
      XP_TotalProfitToday = equity - XP_InitialEquity;
      XP_LastEquityUpdate = current;
   }
}

//+------------------------------------------------------------------+
//| Total and daily drawdown limits                                  |
//+------------------------------------------------------------------+
bool XP_CheckDrawdownLimits()
{
   if(!InpUseEquityProtection) return true;
   if(XP_TradingHalted) return false;

   double equity  = XP_Account.Equity();
   double balance = XP_Account.Balance();
   if(balance <= 0) balance = 0.01;

   double totalDD = (balance - equity) / balance * 100.0;
   if(totalDD >= InpMaxTotalDD)
   {
      PrintFormat("Max total drawdown exceeded: %.2f%% >= %.2f%%. Trading halted.", totalDD, InpMaxTotalDD);
      XP_TradingHalted = true;
      return false;
   }
   else if(totalDD >= (InpMaxTotalDD - InpDDBuffer))
      PrintFormat("Total drawdown near limit: %.2f%% (limit %.2f%%)", totalDD, InpMaxTotalDD);

   double dailyDD = 0.0;
   if(XP_MaxEquityToday > 0) dailyDD = (XP_MaxEquityToday - equity) / XP_MaxEquityToday * 100.0;

   if(XP_MaxEquityToday > 0 && dailyDD >= InpMaxDailyDD)
   {
      PrintFormat("Max daily drawdown exceeded: %.2f%% >= %.2f%%. Trading paused.", dailyDD, InpMaxDailyDD);
      XP_TradingHalted = true;
      return false;
   }
   else if(XP_MaxEquityToday > 0 && dailyDD >= (InpMaxDailyDD - InpDDBuffer))
      PrintFormat("Daily drawdown near limit: %.2f%% (limit %.2f%%)", dailyDD, InpMaxDailyDD);

   return true;
}

//+------------------------------------------------------------------+
//| Daily loss limit                                                 |
//+------------------------------------------------------------------+
bool XP_CheckDailyLossLimit()
{
   if(!InpUseDailyLossLimit) return true;
   if(XP_TradingHalted) return false;

   double todayProfit = XP_TotalProfitToday;
   if(todayProfit >= 0) return true;

   double loss = -todayProfit;
   if(loss >= XP_DailyLossLimit)
   {
      PrintFormat("Daily loss limit reached: %.2f >= %.2f. Trading halted for the day.", loss, XP_DailyLossLimit);
      XP_TradingHalted = true;
      EventSetTimer(3600);
      return false;
   }
   return true;
}

//+------------------------------------------------------------------+
//| Hard stop: close all on drawdown                                 |
//+------------------------------------------------------------------+
void XP_CheckHardStop()
{
   if(!InpUseHardStop) return;

   double equity  = XP_Account.Equity();
   double balance = XP_Account.Balance();
   if(balance <= 0) return;
   double drawdown = (balance - equity) / balance * 100.0;

   if(drawdown >= InpHardStopLevel)
   {
      PrintFormat("Hard stop triggered. Drawdown: %.2f%% >= %.2f%%", drawdown, InpHardStopLevel);
      for(int i = PositionsTotal() - 1; i >= 0; i--)
      {
         ulong ticket = PositionGetTicket(i);
         if(ticket > 0 && PositionGetString(POSITION_SYMBOL) == _Symbol)
            XP_Trade.PositionClose(ticket);
      }
      ExpertRemove();
   }
}


//==================================================================
// SECTION 6 - TRADE MANAGEMENT
//==================================================================
//+------------------------------------------------------------------+
//| Validate SL/TP distances                                         |
//+------------------------------------------------------------------+
bool XP_CheckStopsValid(int type, double currentPrice, double sl, double tp)
{
   if(sl <= 0 && tp <= 0) return true;

   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int stopLevel = (int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   double minDist = stopLevel * point;
   if(minDist <= 0) minDist = 10 * point;

   double atr = XP_GetIndicatorValue(XP_ATRHandle);
   if(atr != EMPTY_VALUE && atr > 0)
      minDist = MathMax(minDist, atr * 0.2);

   if(type == POSITION_TYPE_BUY)
   {
      if(sl > 0 && sl >= currentPrice - minDist)
      {
         PrintFormat("SL=%.5f too close to price=%.5f (min=%.5f)", sl, currentPrice, minDist);
         return false;
      }
      if(tp > 0 && tp <= currentPrice + minDist)
      {
         PrintFormat("TP=%.5f too close to price=%.5f (min=%.5f)", tp, currentPrice, minDist);
         return false;
      }
   }
   else if(type == POSITION_TYPE_SELL)
   {
      if(sl > 0 && sl <= currentPrice + minDist)
      {
         PrintFormat("SL=%.5f too close to price=%.5f (min=%.5f)", sl, currentPrice, minDist);
         return false;
      }
      if(tp > 0 && tp >= currentPrice - minDist)
      {
         PrintFormat("TP=%.5f too close to price=%.5f (min=%.5f)", tp, currentPrice, minDist);
         return false;
      }
   }
   return true;
}

//+------------------------------------------------------------------+
//| Adaptive trailing stop                                           |
//+------------------------------------------------------------------+
void XP_ApplyTrailing()
{
   if(!InpUseTrailing) return;
   if(!XP_MarketOpen()) return;   // skip modifications when the market is closed
   if(TimeCurrent() - XP_LastTrailTime < 30) return;

   double atr = XP_GetIndicatorValue(XP_ATRHandle);
   if(atr <= 0)
   {
      Print("Failed to read ATR for trailing");
      return;
   }

   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double currentAsk = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double currentBid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double chaos = XP_CalculateChaos();

   double trailMultiplier = XP_BaseTrailATRMult +
      (InpMaxTrailATRMult - XP_BaseTrailATRMult) *
      MathMin(1.0, chaos * InpChaosSensitivity + XP_State * InpStateInfluence / XP_States);

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket <= 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;

      double currentSl    = PositionGetDouble(POSITION_SL);
      double positionOpen = PositionGetDouble(POSITION_PRICE_OPEN);
      int    positionType = (int)PositionGetInteger(POSITION_TYPE);
      double currentPrice = (positionType == POSITION_TYPE_BUY) ? currentBid : currentAsk;

      double activationLevel = (positionType == POSITION_TYPE_BUY) ?
         positionOpen + InpTrailActivation * atr :
         positionOpen - InpTrailActivation * atr;

      double proposedSl = (positionType == POSITION_TYPE_BUY) ?
         currentBid - trailMultiplier * atr :
         currentAsk + trailMultiplier * atr;

      double stateCorrection = (XP_State - XP_States/2.0) * atr * 0.05;
      proposedSl += (positionType == POSITION_TYPE_BUY) ? -stateCorrection : stateCorrection;

      double minAllowedDistance = MathMax(
         SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * point, atr * 0.3);

      if((positionType == POSITION_TYPE_BUY  && currentPrice < activationLevel) ||
         (positionType == POSITION_TYPE_SELL && currentPrice > activationLevel))
         continue;

      double minChange = 3 * point;
      if(MathAbs(proposedSl - currentSl) < minChange) continue;

      bool canModify = false;
      if(positionType == POSITION_TYPE_BUY)
      {
         if(proposedSl > currentSl && proposedSl < currentBid - minAllowedDistance) canModify = true;
      }
      else
      {
         if(proposedSl < currentSl && proposedSl > currentAsk + minAllowedDistance) canModify = true;
      }

      if(!canModify) continue;

      double priceChange = MathAbs(currentPrice - positionOpen);
      if(priceChange < atr * 0.3) continue;

      if(!XP_CheckStopsValid(positionType, currentPrice, proposedSl, PositionGetDouble(POSITION_TP)))
      {
         Print("New SL failed validation. Modification cancelled.");
         continue;
      }

      if(MathAbs(proposedSl - XP_LastTrailSl) < 10*point && MathAbs(currentPrice - XP_LastTrailPrice) < 10*point)
         continue;

      double newSl = XP_NormalizePrice(proposedSl);
      if(XP_Trade.PositionModify(ticket, newSl, PositionGetDouble(POSITION_TP)))
      {
         XP_State = (XP_State + 1) % XP_States;
         PrintFormat("Trailing updated: %s, SL %.5f -> %.5f",
                     EnumToString((ENUM_POSITION_TYPE)positionType), currentSl, newSl);
         XP_LastTrailTime  = TimeCurrent();
         XP_LastTrailSl    = newSl;
         XP_LastTrailPrice = currentPrice;
      }
      else
      {
         uint errorCode = GetLastError();
         Print("Trailing modify error: ", XP_ErrorDescription((int)errorCode));
         if(errorCode == 4756)
         {
            double adjustedSl = XP_NormalizePrice(newSl + ((positionType==POSITION_TYPE_BUY) ? -point : point));
            if(XP_CheckStopsValid(positionType, currentPrice, adjustedSl, PositionGetDouble(POSITION_TP)))
            {
               if(XP_Trade.PositionModify(ticket, adjustedSl, PositionGetDouble(POSITION_TP)))
               {
                  Print("SL adjusted by 1 point");
                  XP_LastTrailTime = TimeCurrent();
               }
            }
         }
      }
   }
}


//==================================================================
// SECTION 7 - EVENT HANDLERS
//==================================================================
//+------------------------------------------------------------------+
//| OnInit - initialization                                          |
//+------------------------------------------------------------------+
int OnInit()
{
   MathSrand(GetTickCount());

   XP_States            = InpStates;
   XP_StateDecayRate    = InpStateDecayRate;
   XP_StateUpdateRate   = InpStateUpdateRate;
   XP_StateMemoryFactor = InpStateMemoryFactor;
   XP_RSIPeriod         = InpRSIPeriod;
   XP_ADXPeriod         = InpADXPeriod;
   XP_ATRPeriod         = InpATRPeriod;
   XP_FastMAPeriod      = InpFastMAPeriod;
   XP_SlowMAPeriod      = InpSlowMAPeriod;
   XP_BaseRisk          = InpBaseRisk;
   XP_ATRMultSL         = InpATRMultSL;
   XP_TPtoSL            = InpTPtoSL;
   XP_MinBars           = InpMinBarsBetweenTrades;
   XP_BaseTrailATRMult  = InpBaseTrailATRMult;
   XP_ChaosThreshold    = InpChaosThreshold;
   XP_CurrentRisk       = InpBaseRisk;

   ArrayResize(XP_WaveFunction, XP_States);
   ArrayResize(XP_StateReturns, XP_States);
   ArrayResize(XP_PositionMatrix, XP_States);
   ArrayInitialize(XP_WaveFunction, 1.0/XP_States);
   ArrayInitialize(XP_StateReturns, 0);

   if(InpAutoOptimize && MQLInfoInteger(MQL_TESTER))
      XP_LoadOptimizedParams();

   XP_AutoCorrectParameters();

   // Apply per-symbol preset (only when the master toggle is enabled)
   string symbol = Symbol();
   if(InpUseSymbolPresets && (symbol == "XAUUSD" || symbol == "GOLD"))
   {
      XP_ATRMultSL        = InpGoldATRMultSL;
      XP_TPtoSL           = InpGoldTPtoSL;
      XP_BaseRisk         = InpGoldBaseRisk;
      XP_MinBars          = InpGoldMinBars;
      XP_BaseTrailATRMult = InpGoldTrailATRMult;
      XP_ChaosThreshold   = InpGoldChaosThreshold;
   }
   else if(InpUseSymbolPresets && (symbol == "XAGUSD" || symbol == "SILVER"))
   {
      XP_ATRMultSL        = InpSilverATRMultSL;
      XP_TPtoSL           = InpSilverTPtoSL;
      XP_BaseRisk         = InpSilverBaseRisk;
      XP_MinBars          = InpSilverMinBars;
      XP_BaseTrailATRMult = InpSilverTrailATRMult;
      XP_ChaosThreshold   = InpSilverChaosThreshold;
   }

   XP_InitSignalEngine();

   XP_LastTradeTime   = 0;
   XP_LossCounter     = 0;
   XP_LastCacheTime   = 0;
   XP_LastTrailTime   = 0;
   XP_LastTrailSl     = 0;
   XP_LastTrailPrice  = 0;
   XP_MarginErrorCount= 0;
   XP_LastErrorTime   = 0;

   XP_UpdateDailyEquity();

   XP_MicroMode = InpMicroMode;
   if(XP_Account.Equity() < 500)
   {
      XP_MicroMode = true;
      Print("Micro account mode auto-enabled: equity < 500");
   }

   Print("EA initialized on ", symbol);
   PrintFormat("Equity: %.2f", XP_Account.Equity());
   PrintFormat("Base risk: %.1f%%, Max risk: %.1f%%", XP_BaseRisk, InpMaxRisk);

   XP_IsFirstRun = true;
   XP_OptimizationCompleted = false;

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| OnTick - main trading loop                                       |
//+------------------------------------------------------------------+
void OnTick()
{
   if(XP_TradingHalted)
   {
      static datetime lastAlert = 0;
      if(TimeCurrent() - lastAlert > 3600)
      {
         Print("Trading paused by protection system");
         lastAlert = TimeCurrent();
      }
      return;
   }

   if(XP_IsFirstRun && MQLInfoInteger(MQL_TESTER) && InpAutoOptimize && !XP_OptimizationCompleted)
   {
      XP_RunAutoOptimization();
      XP_IsFirstRun = false;
      return;
   }

   XP_ApplyTrailing();
   XP_CheckHardStop();

   static datetime lastBarTime = 0;
   datetime currentBarTime = iTime(_Symbol, InpTimeframe, 0);
   if(currentBarTime == lastBarTime) return;
   lastBarTime = currentBarTime;

   XP_UpdateDailyEquity();

   if(!XP_CheckDrawdownLimits()) return;
   if(!XP_CheckDailyLossLimit()) return;

   int signal = XP_GenerateTradeSignal();
   if(signal < 0) return;

   if(TimeCurrent() - XP_LastTradeTime < XP_MinBars * PeriodSeconds(InpTimeframe)) return;

   double atr = XP_LastATR;
   double price = (signal == ORDER_TYPE_BUY) ?
      SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);

   double sl = (signal == ORDER_TYPE_BUY) ?
      price - atr * XP_ATRMultSL : price + atr * XP_ATRMultSL;
   double tp = (signal == ORDER_TYPE_BUY) ?
      price + atr * XP_TPtoSL : price - atr * XP_TPtoSL;

   double minTpDistance = atr * 0.5;
   if(MathAbs(price - tp) < minTpDistance)
      tp = (signal == ORDER_TYPE_BUY) ? price + minTpDistance : price - minTpDistance;

   price = XP_NormalizePrice(price);
   sl    = XP_NormalizePrice(sl);
   tp    = XP_NormalizePrice(tp);

   double checkPrice = (signal == ORDER_TYPE_BUY) ?
      SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);

   if(!XP_CheckStopsValid(signal == ORDER_TYPE_BUY ? POSITION_TYPE_BUY : POSITION_TYPE_SELL, checkPrice, sl, tp))
   {
      Print("Invalid SL/TP. Trade rejected.");
      return;
   }

   XP_CurrentRisk = XP_CalculateAdaptiveRisk();
   double slDistance = MathAbs(price - sl);
   double lotSize = XP_CalculateLotSize(XP_CurrentRisk, price, slDistance);

   if(lotSize < 0)
   {
      Print("Trade impossible: not enough margin for min lot");
      XP_MarginErrorCount++;
      XP_LastErrorTime = TimeCurrent();
      if(XP_MarginErrorCount >= 3)
      {
         XP_TradingHalted = true;
         Print("Trading paused: 3 consecutive margin errors");
         EventSetTimer(3600);
      }
      return;
   }
   else XP_MarginErrorCount = 0;

   if(XP_MicroMode)
   {
      double maxMicroLot = XP_Account.FreeMargin() / (price * 0.01);
      if(lotSize > maxMicroLot)
      {
         lotSize = MathMin(lotSize, maxMicroLot);
         PrintFormat("Micro account lot correction: %.2f", lotSize);
      }
   }

   ENUM_ORDER_TYPE orderType = (signal == ORDER_TYPE_BUY) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;

   if(!XP_CheckMoneyForTrade(_Symbol, lotSize, orderType))
   {
      Print("Money check failed. Trade rejected.");
      return;
   }

   if(!XP_CheckPositionSize(lotSize, price))
   {
      Print("Position size over limit. Trade rejected.");
      return;
   }

   bool success = false;
   if(signal == ORDER_TYPE_BUY)
      success = XP_Trade.Buy(lotSize, _Symbol, price, sl, tp, "QGST Buy");
   else
      success = XP_Trade.Sell(lotSize, _Symbol, price, sl, tp, "QGST Sell");

   if(success)
   {
      XP_LastTradeTime = TimeCurrent();
      PrintFormat("Trade: %s Lot: %.2f Risk: %.1f%% ATR: %.2f",
                  EnumToString(orderType), lotSize, XP_CurrentRisk, atr);
   }
   else
   {
      uint errorCode = GetLastError();
      Print("Order error: ", errorCode, " - ", XP_ErrorDescription((int)errorCode));
      XP_CurrentRisk = MathMax(0.1, XP_CurrentRisk * InpRiskDecreaseFactor);
   }

   XP_IsFirstRun = false;
}

//+------------------------------------------------------------------+
//| OnTimer - protection auto-recovery                               |
//+------------------------------------------------------------------+
void OnTimer()
{
   if(XP_TradingHalted)
   {
      XP_CurrentRisk = MathMax(0.1, XP_CurrentRisk * 0.7);
      XP_TradingHalted = false;
      XP_MarginErrorCount = 0;
      Print("Trading resumed. Risk reduced to ", XP_CurrentRisk, "%");
   }
}

//+------------------------------------------------------------------+
//| OnTrade - adaptive update after deals                            |
//+------------------------------------------------------------------+
void OnTrade()
{
   if(!HistorySelect(0, TimeCurrent())) return;
   int total = HistoryDealsTotal();
   if(total <= 0) return;

   ulong ticket = HistoryDealGetTicket(total-1);
   if(ticket == 0) return;

   double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT);
   int    type   = (int)HistoryDealGetInteger(ticket, DEAL_TYPE);
   int    signal = (type == DEAL_TYPE_BUY) ? 1 : -1;

   XP_UpdateLearning(signal, profit);

   if(XP_State >= 0 && XP_State < ArraySize(XP_StateReturns))
      XP_StateReturns[XP_State] = XP_StateReturns[XP_State] * XP_StateMemoryFactor + profit * XP_StateUpdateRate;

   if(profit < 0)
   {
      XP_LossCounter++;
      if(XP_LossCounter >= InpMaxConsecutiveLosses)
         XP_CurrentRisk = MathMax(0.1, XP_CurrentRisk * InpRiskDecreaseFactor);
   }
   else
   {
      XP_LossCounter = 0;
      XP_CurrentRisk = MathMin(InpMaxRisk, XP_CurrentRisk * InpRiskIncreaseFactor);
   }

   XP_UpdateDailyEquity();
}

//+------------------------------------------------------------------+
//| OnTester - 70/30 in-sample / out-of-sample score                 |
//+------------------------------------------------------------------+
double OnTester()
{
   HistorySelect(0, TimeCurrent());
   int total_deals = HistoryDealsTotal();
   if(total_deals <= 0) return 0;

   datetime first_deal_time = (datetime)HistoryDealGetInteger(HistoryDealGetTicket(0), DEAL_TIME);
   datetime last_deal_time  = (datetime)HistoryDealGetInteger(HistoryDealGetTicket(total_deals-1), DEAL_TIME);

   datetime IS_end    = first_deal_time + (datetime)((last_deal_time - first_deal_time) * 0.7);
   datetime OOS_start = IS_end + 1;
   datetime OOS_end   = last_deal_time;

   double deposit = 10000;
   double balance = deposit;
   double max_balance = deposit;

   double IS_dd = 0, IS_profit = 0, IS_positive = 0, IS_negative = 0;
   int    IS_trades = 0;

   double OOS_dd = 0, OOS_profit = 0, OOS_positive = 0, OOS_negative = 0;
   int    OOS_trades = 0;

   for(int i = 0; i < total_deals; i++)
   {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket == 0) continue;

      datetime time = (datetime)HistoryDealGetInteger(ticket, DEAL_TIME);
      double total = HistoryDealGetDouble(ticket, DEAL_PROFIT)
                   + HistoryDealGetDouble(ticket, DEAL_COMMISSION)
                   + HistoryDealGetDouble(ticket, DEAL_SWAP);

      balance += total;
      if(balance > max_balance) max_balance = balance;
      double dd = max_balance - balance;

      if(time <= IS_end)
      {
         IS_trades++;
         IS_profit += total;
         if(total > 0) IS_positive += total; else IS_negative -= total;
         if(dd > IS_dd) IS_dd = dd;
      }
      else if(time >= OOS_start && time <= OOS_end)
      {
         OOS_trades++;
         OOS_profit += total;
         if(total > 0) OOS_positive += total; else OOS_negative -= total;
         if(dd > OOS_dd) OOS_dd = dd;
      }
   }

   double IS_rf  = (IS_dd > 0) ? IS_profit / IS_dd : IS_profit * 100;
   double IS_win = (IS_positive + IS_negative > 0) ? IS_positive/(IS_positive+IS_negative) : 0;
   double IS_avg_win  = (IS_trades > 0 && IS_win > 0) ? IS_positive/(IS_trades * IS_win) : 0;
   double IS_avg_loss = (IS_trades > 0 && (1-IS_win) > 0) ? IS_negative/(IS_trades * (1-IS_win)) : 0;
   double IS_payoff = (IS_avg_loss != 0) ? IS_avg_win/IS_avg_loss : 10;

   double OOS_pf  = (OOS_negative > 0) ? OOS_positive / OOS_negative : 100;
   double OOS_rf  = (OOS_dd > 0) ? OOS_profit / OOS_dd : OOS_profit * 100;
   double OOS_win = (OOS_positive + OOS_negative > 0) ? OOS_positive/(OOS_positive+OOS_negative) : 0;
   double OOS_avg_win  = (OOS_trades > 0 && OOS_win > 0) ? OOS_positive/(OOS_trades * OOS_win) : 0;
   double OOS_avg_loss = (OOS_trades > 0 && (1-OOS_win) > 0) ? OOS_negative/(OOS_trades * (1-OOS_win)) : 0;
   double OOS_payoff = (OOS_avg_loss != 0) ? OOS_avg_win/OOS_avg_loss : 10;

   if(OOS_trades < 20) return -1e9;
   if(OOS_profit <= 0) return -1e8;
   if(IS_profit  <= 0) return -1e7;
   if(OOS_rf     < 1.0) return -1e6;

   double score = 0;
   score += 15.0 * MathLog(OOS_rf + 1);
   score += 10.0 * MathLog(OOS_pf + 1);
   score += 7.5  * MathLog(OOS_payoff + 1);

   double consistency = 1.0;
   if(IS_win    > 0) consistency *= 0.7 + 0.3 * MathMin(OOS_win/IS_win, 1.5);
   if(IS_payoff > 0) consistency *= 0.7 + 0.3 * MathMin(OOS_payoff/IS_payoff, 1.5);
   if(IS_rf     > 0) consistency *= 0.7 + 0.3 * MathMin(OOS_rf/IS_rf, 1.5);
   score += 30.0 * consistency;

   double stability = 1.0;
   stability -= 0.2 * MathAbs(IS_win - OOS_win);
   stability -= 0.2 * MathAbs(IS_payoff - OOS_payoff);
   score += 20.0 * MathMax(0, stability);

   double risk_penalty = 1.0 - 0.3 * MathMin(1.0, OOS_dd / deposit);
   score *= MathMax(0.5, risk_penalty);

   score *= 1.0 + 0.1 * MathLog(1 + OOS_trades);

   return score;
}

//+------------------------------------------------------------------+
//| OnDeinit - cleanup                                               |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   string fileName = "QGST_Weights_" + _Symbol + ".txt";
   int handle = FileOpen(fileName, FILE_WRITE|FILE_TXT|FILE_ANSI|FILE_COMMON);
   if(handle != INVALID_HANDLE)
   {
      for(int i = 0; i < 6; i++)
         FileWrite(handle, DoubleToString(XP_AIWeights[i], 8));
      FileClose(handle);
   }

   if(XP_OptimizationCompleted && InpAutoOptimize && MQLInfoInteger(MQL_TESTER))
      XP_SaveOptimizedParams();

   if(XP_RSIHandle    != INVALID_HANDLE) IndicatorRelease(XP_RSIHandle);
   if(XP_ADXHandle    != INVALID_HANDLE) IndicatorRelease(XP_ADXHandle);
   if(XP_ATRHandle    != INVALID_HANDLE) IndicatorRelease(XP_ATRHandle);
   if(XP_MAFastHandle != INVALID_HANDLE) IndicatorRelease(XP_MAFastHandle);
   if(XP_MASlowHandle != INVALID_HANDLE) IndicatorRelease(XP_MASlowHandle);

   EventKillTimer();
}

//+------------------------------------------------------------------+

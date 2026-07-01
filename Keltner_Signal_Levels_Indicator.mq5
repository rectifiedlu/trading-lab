// Keltner channel signal levels for visual checks.
// Center is EMA(close, Length); channel is center +/- Mult * ATR(Length).
#property indicator_chart_window
#property indicator_plots 5
#property indicator_buffers 5

#property indicator_label1 "Keltner Upper"
#property indicator_type1 DRAW_LINE
#property indicator_color1 clrDeepSkyBlue
#property indicator_width1 1

#property indicator_label2 "Keltner Middle"
#property indicator_type2 DRAW_LINE
#property indicator_color2 clrSilver
#property indicator_width2 1

#property indicator_label3 "Keltner Lower"
#property indicator_type3 DRAW_LINE
#property indicator_color3 clrOrange
#property indicator_width3 1

#property indicator_label4 "Invert Short"
#property indicator_type4 DRAW_ARROW
#property indicator_color4 clrTomato
#property indicator_width4 2

#property indicator_label5 "Invert Long"
#property indicator_type5 DRAW_ARROW
#property indicator_color5 clrLime
#property indicator_width5 2

input int Length = 20;
input double Mult = 1.5;
input bool InvertSignals = true;

double UpperBuf[];
double MiddleBuf[];
double LowerBuf[];
double ShortArrowBuf[];
double LongArrowBuf[];

int AtrHandle = INVALID_HANDLE;
int EmaHandle = INVALID_HANDLE;

int OnInit()
{
   SetIndexBuffer(0, UpperBuf, INDICATOR_DATA);
   SetIndexBuffer(1, MiddleBuf, INDICATOR_DATA);
   SetIndexBuffer(2, LowerBuf, INDICATOR_DATA);
   SetIndexBuffer(3, ShortArrowBuf, INDICATOR_DATA);
   SetIndexBuffer(4, LongArrowBuf, INDICATOR_DATA);

   PlotIndexSetInteger(3, PLOT_ARROW, 234);
   PlotIndexSetInteger(4, PLOT_ARROW, 233);
   PlotIndexSetDouble(3, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetDouble(4, PLOT_EMPTY_VALUE, EMPTY_VALUE);

   AtrHandle = iATR(_Symbol, _Period, Length);
   EmaHandle = iMA(_Symbol, _Period, Length, 0, MODE_EMA, PRICE_CLOSE);
   if(AtrHandle == INVALID_HANDLE || EmaHandle == INVALID_HANDLE)
      return INIT_FAILED;

   IndicatorSetString(INDICATOR_SHORTNAME, "Keltner Signal Levels");
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   if(AtrHandle != INVALID_HANDLE) IndicatorRelease(AtrHandle);
   if(EmaHandle != INVALID_HANDLE) IndicatorRelease(EmaHandle);
}

int OnCalculate(
   const int rates_total,
   const int prev_calculated,
   const datetime &time[],
   const double &open[],
   const double &high[],
   const double &low[],
   const double &close[],
   const long &tick_volume[],
   const long &volume[],
   const int &spread[]
)
{
   if(rates_total <= Length + 2)
      return 0;

   static double atr[];
   static double ema[];
   ArrayResize(atr, rates_total);
   ArrayResize(ema, rates_total);
   ArraySetAsSeries(atr, true);
   ArraySetAsSeries(ema, true);
   ArraySetAsSeries(UpperBuf, true);
   ArraySetAsSeries(MiddleBuf, true);
   ArraySetAsSeries(LowerBuf, true);
   ArraySetAsSeries(ShortArrowBuf, true);
   ArraySetAsSeries(LongArrowBuf, true);

   if(CopyBuffer(AtrHandle, 0, 0, rates_total, atr) <= 0)
      return prev_calculated;
   if(CopyBuffer(EmaHandle, 0, 0, rates_total, ema) <= 0)
      return prev_calculated;

   int limit = rates_total - Length - 1;
   for(int i = limit; i >= 0; --i)
   {
      double middle = ema[i];
      double channel = Mult * atr[i];
      double upper = middle + channel;
      double lower = middle - channel;

      UpperBuf[i] = upper;
      MiddleBuf[i] = middle;
      LowerBuf[i] = lower;
      ShortArrowBuf[i] = EMPTY_VALUE;
      LongArrowBuf[i] = EMPTY_VALUE;

      bool break_up = close[i] > upper;
      bool break_down = close[i] < lower;

      if(InvertSignals)
      {
         if(break_up) ShortArrowBuf[i] = high[i] + 20 * _Point;
         if(break_down) LongArrowBuf[i] = low[i] - 20 * _Point;
      }
      else
      {
         if(break_up) LongArrowBuf[i] = low[i] - 20 * _Point;
         if(break_down) ShortArrowBuf[i] = high[i] + 20 * _Point;
      }
   }

   for(int i = rates_total - 1; i > limit; --i)
   {
      UpperBuf[i] = EMPTY_VALUE;
      MiddleBuf[i] = EMPTY_VALUE;
      LowerBuf[i] = EMPTY_VALUE;
      ShortArrowBuf[i] = EMPTY_VALUE;
      LongArrowBuf[i] = EMPTY_VALUE;
   }

   return rates_total;
}

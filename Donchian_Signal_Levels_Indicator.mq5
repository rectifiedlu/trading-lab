// Donchian signal levels for visual checks.
// Upper/lower use the previous Length closed bars, excluding the current bar.
#property indicator_chart_window
#property indicator_plots 4
#property indicator_buffers 4

#property indicator_label1 "Donchian Upper"
#property indicator_type1 DRAW_LINE
#property indicator_color1 clrDeepSkyBlue
#property indicator_width1 1

#property indicator_label2 "Donchian Lower"
#property indicator_type2 DRAW_LINE
#property indicator_color2 clrOrange
#property indicator_width2 1

#property indicator_label3 "Invert Short"
#property indicator_type3 DRAW_ARROW
#property indicator_color3 clrTomato
#property indicator_width3 2

#property indicator_label4 "Invert Long"
#property indicator_type4 DRAW_ARROW
#property indicator_color4 clrLime
#property indicator_width4 2

input int Length = 16;
input bool InvertSignals = true;

double UpperBuf[];
double LowerBuf[];
double ShortArrowBuf[];
double LongArrowBuf[];

int OnInit()
{
   SetIndexBuffer(0, UpperBuf, INDICATOR_DATA);
   SetIndexBuffer(1, LowerBuf, INDICATOR_DATA);
   SetIndexBuffer(2, ShortArrowBuf, INDICATOR_DATA);
   SetIndexBuffer(3, LongArrowBuf, INDICATOR_DATA);

   PlotIndexSetDouble(0, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetDouble(1, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetInteger(2, PLOT_ARROW, 234);
   PlotIndexSetInteger(3, PLOT_ARROW, 233);
   PlotIndexSetDouble(2, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetDouble(3, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetInteger(0, PLOT_DRAW_BEGIN, Length + 1);
   PlotIndexSetInteger(1, PLOT_DRAW_BEGIN, Length + 1);

   IndicatorSetString(INDICATOR_SHORTNAME, "Donchian Signal Levels");
   return INIT_SUCCEEDED;
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

   ArraySetAsSeries(time, true);
   ArraySetAsSeries(open, true);
   ArraySetAsSeries(high, true);
   ArraySetAsSeries(low, true);
   ArraySetAsSeries(close, true);
   ArraySetAsSeries(UpperBuf, true);
   ArraySetAsSeries(LowerBuf, true);
   ArraySetAsSeries(ShortArrowBuf, true);
   ArraySetAsSeries(LongArrowBuf, true);

   for(int i = 0; i < rates_total; ++i)
   {
      UpperBuf[i] = EMPTY_VALUE;
      LowerBuf[i] = EMPTY_VALUE;
      ShortArrowBuf[i] = EMPTY_VALUE;
      LongArrowBuf[i] = EMPTY_VALUE;
   }

   int limit = rates_total - Length - 2;
   for(int i = limit; i >= 0; --i)
   {
      double upper = high[i + 1];
      double lower = low[i + 1];
      for(int k = i + 1; k <= i + Length; ++k)
      {
         if(high[k] > upper) upper = high[k];
         if(low[k] < lower) lower = low[k];
      }

      UpperBuf[i] = upper;
      LowerBuf[i] = lower;

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

   return rates_total;
}

#property indicator_chart_window
#property indicator_plots 2
#property indicator_buffers 2

#property indicator_label1 "SAR"
#property indicator_type1 DRAW_ARROW
#property indicator_color1 clrOrange
#property indicator_width1 2

#property indicator_label2 "NextBarSAR"
#property indicator_type2 DRAW_ARROW
#property indicator_color2 clrAqua
#property indicator_width2 2

input double StartAF = 0.03;
input double IncrementAF = 0.03;
input double MaximumAF = 0.20;

double SarBuffer[];
double NextSarBuffer[];

int OnInit()
{
   SetIndexBuffer(0, SarBuffer, INDICATOR_DATA);
   SetIndexBuffer(1, NextSarBuffer, INDICATOR_DATA);

   PlotIndexSetInteger(0, PLOT_ARROW, 159);
   PlotIndexSetInteger(1, PLOT_ARROW, 159);

   PlotIndexSetDouble(0, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetDouble(1, PLOT_EMPTY_VALUE, EMPTY_VALUE);

   IndicatorSetString(INDICATOR_SHORTNAME,
      StringFormat("PSAR Next (%.3f, %.3f, %.3f)", StartAF, IncrementAF, MaximumAF));
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
   if(rates_total < 3)
      return 0;

   ArrayInitialize(SarBuffer, EMPTY_VALUE);
   ArrayInitialize(NextSarBuffer, EMPTY_VALUE);

   bool uptrend = false;
   double ep = 0.0;
   double af = StartAF;
   double nextBarSAR = 0.0;

   // Arrays in OnCalculate are series: index 0 is newest.
   // Work oldest -> newest to mirror TradingView/Python logic.
   for(int i = rates_total - 2; i >= 0; --i)
   {
      bool firstTrendBar = false;
      double sar = nextBarSAR;

      if(i == rates_total - 2)
      {
         double lowPrev = low[i + 1];
         double highPrev = high[i + 1];
         double prevSAR;
         double prevEP;

         if(close[i] > close[i + 1])
         {
            uptrend = true;
            ep = high[i];
            prevSAR = lowPrev;
            prevEP = high[i];
         }
         else
         {
            uptrend = false;
            ep = low[i];
            prevSAR = highPrev;
            prevEP = low[i];
         }

         firstTrendBar = true;
         sar = prevSAR + StartAF * (prevEP - prevSAR);
      }

      if(uptrend)
      {
         if(sar > low[i])
         {
            firstTrendBar = true;
            uptrend = false;
            sar = MathMax(ep, high[i]);
            ep = low[i];
            af = StartAF;
         }
      }
      else
      {
         if(sar < high[i])
         {
            firstTrendBar = true;
            uptrend = true;
            sar = MathMin(ep, low[i]);
            ep = high[i];
            af = StartAF;
         }
      }

      if(!firstTrendBar)
      {
         if(uptrend)
         {
            if(high[i] > ep)
            {
               ep = high[i];
               af = MathMin(af + IncrementAF, MaximumAF);
            }
         }
         else
         {
            if(low[i] < ep)
            {
               ep = low[i];
               af = MathMin(af + IncrementAF, MaximumAF);
            }
         }
      }

      if(uptrend)
      {
         sar = MathMin(sar, low[i + 1]);
         if(i + 2 < rates_total)
            sar = MathMin(sar, low[i + 2]);
      }
      else
      {
         sar = MathMax(sar, high[i + 1]);
         if(i + 2 < rates_total)
            sar = MathMax(sar, high[i + 2]);
      }

      nextBarSAR = sar + af * (ep - sar);
      SarBuffer[i] = sar;
      NextSarBuffer[i] = nextBarSAR;
   }

   return rates_total;
}

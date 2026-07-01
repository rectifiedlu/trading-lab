// Volty levels overlay indicator for MT5.
// Copy into MQL5/Indicators, compile, then attach to the chart.
#property strict
#property indicator_chart_window
#property indicator_plots 4
#property indicator_buffers 4

#property indicator_label1 "Volty Entry Upper"
#property indicator_type1 DRAW_LINE
#property indicator_color1 clrLime
#property indicator_style1 STYLE_SOLID
#property indicator_width1 2

#property indicator_label2 "Volty Entry Lower"
#property indicator_type2 DRAW_LINE
#property indicator_color2 clrRed
#property indicator_style2 STYLE_SOLID
#property indicator_width2 2

#property indicator_label3 "Volty Hold Close Upper"
#property indicator_type3 DRAW_LINE
#property indicator_color3 clrOrange
#property indicator_style3 STYLE_DOT
#property indicator_width3 1

#property indicator_label4 "Volty Hold Close Lower"
#property indicator_type4 DRAW_LINE
#property indicator_color4 clrOrange
#property indicator_style4 STYLE_DOT
#property indicator_width4 1

input int    Length          = 4;
input double ATRMult         = 0.73;
input double HoldCloseMult   = 0.70;
input double MinATRPoints    = 0.0;

double EntryUpper[];
double EntryLower[];
double HoldUpper[];
double HoldLower[];

double TrueRangeAt(
   const int i,
   const double &high[],
   const double &low[],
   const double &close[],
   const int rates_total
)
{
   if(i + 1 >= rates_total)
      return high[i] - low[i];
   double prevClose = close[i + 1];
   return MathMax(high[i] - low[i], MathMax(MathAbs(high[i] - prevClose), MathAbs(low[i] - prevClose)));
}

int OnInit()
{
   if(Length < 1)
      return INIT_PARAMETERS_INCORRECT;
   SetIndexBuffer(0, EntryUpper, INDICATOR_DATA);
   SetIndexBuffer(1, EntryLower, INDICATOR_DATA);
   SetIndexBuffer(2, HoldUpper, INDICATOR_DATA);
   SetIndexBuffer(3, HoldLower, INDICATOR_DATA);

   ArraySetAsSeries(EntryUpper, true);
   ArraySetAsSeries(EntryLower, true);
   ArraySetAsSeries(HoldUpper, true);
   ArraySetAsSeries(HoldLower, true);

   IndicatorSetString(INDICATOR_SHORTNAME, "Volty Levels");
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
   if(rates_total <= Length + 1)
      return 0;

   ArraySetAsSeries(high, true);
   ArraySetAsSeries(low, true);
   ArraySetAsSeries(close, true);

   EntryUpper[rates_total - 1] = EMPTY_VALUE;
   EntryLower[rates_total - 1] = EMPTY_VALUE;
   HoldUpper[rates_total - 1] = EMPTY_VALUE;
   HoldLower[rates_total - 1] = EMPTY_VALUE;

   int limit = rates_total - Length - 1;
   for(int i = limit; i >= 1; --i)
   {
      double sumTR = 0.0;
      bool ok = true;
      for(int j = 0; j < Length; ++j)
      {
         int idx = i + j;
         if(idx + 1 >= rates_total)
         {
            ok = false;
            break;
         }
         sumTR += TrueRangeAt(idx, high, low, close, rates_total);
      }

      if(!ok)
      {
         EntryUpper[i] = EMPTY_VALUE;
         EntryLower[i] = EMPTY_VALUE;
         HoldUpper[i] = EMPTY_VALUE;
         HoldLower[i] = EMPTY_VALUE;
         continue;
      }

      double rawATR = sumTR / Length;
      double minATR = MinATRPoints * _Point;
      bool atrPass = MinATRPoints <= 0.0 || rawATR >= minATR;
      double entryDist = rawATR * ATRMult;
      double holdDist = rawATR * HoldCloseMult;

      int activeBar = i - 1;
      EntryUpper[activeBar] = atrPass ? close[i] + entryDist : EMPTY_VALUE;
      EntryLower[activeBar] = atrPass ? close[i] - entryDist : EMPTY_VALUE;
      HoldUpper[activeBar] = close[i] + holdDist;
      HoldLower[activeBar] = close[i] - holdDist;
   }

   return rates_total;
}

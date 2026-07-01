//+------------------------------------------------------------------+
//|                                                    InsideBar.mq5 |
//|                                                      reza rahmad |
//|                                             rezarahmad@gmail.com |
//+------------------------------------------------------------------+
#property copyright "reza rahmad"
#property link      "rezarahmad@gmail.com"
#property version   "1.00"
#property indicator_chart_window
#property indicator_buffers 0
#property indicator_plots   0

//--- Input parameters (Basic)
input int      BarsToKeep         = 50;       // Number of candles backward to keep rectangles
input int      ForwardBars        = 5;        // Rectangle extension (bars ahead from big candle)
input bool     RectangleFill       = true;    // Fill rectangle with color

//--- Color inputs
input bool     AutoColorByDirection = false;  // Auto color based on small candle direction
input color    BullishColor        = clrGreen;  // Color for bullish signal
input color    BearishColor        = clrRed;    // Color for bearish signal
input color    NeutralColor        = clrBlue;   // Color for neutral signal (border when auto off)
input color    RectangleFillColor   = clrBlue;  // Fill color when auto color is off

//--- Premium features
input bool     EnableAlert         = true;    // Enable notifications (popup, push, sound)
input bool     EnableLabel         = true;    // Show text labels on chart
input bool     LabelOnlyOnNewBar   = true;    // Show label only on latest pattern
input bool     FilterDirection     = false;   // Show only bullish/bearish signals (ignore neutral)

//--- Global variables
string          Prefix = "IBRect_";
datetime        lastBarTime = 0;
int             actualBarsToKeep;
int             actualForwardBars;
bool            alertSentForBar[];

//+------------------------------------------------------------------+
//| Custom indicator initialization function                          |
//+------------------------------------------------------------------+
int OnInit()
  {
   actualBarsToKeep = (BarsToKeep < 1) ? 1 : BarsToKeep;
   actualForwardBars = (ForwardBars < 1) ? 1 : ForwardBars;
   ArrayResize(alertSentForBar, 0);
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Custom indicator deinitialization function                        |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   ObjectsDeleteAll(0, Prefix);
  }

//+------------------------------------------------------------------+
//| Delete rectangles older than BarsToKeep bars                      |
//+------------------------------------------------------------------+
void DeleteOldRectangles(datetime currentTime)
  {
   int secondsPerBar = PeriodSeconds();
   datetime oldestTime = currentTime - actualBarsToKeep * secondsPerBar;

   for(int i = ObjectsTotal(0, 0, OBJ_RECTANGLE) - 1; i >= 0; i--)
     {
      string objName = ObjectName(0, i, 0, OBJ_RECTANGLE);
      if(StringFind(objName, Prefix) == 0)
        {
         datetime leftTime = (datetime)ObjectGetInteger(0, objName, OBJPROP_TIME, 0);
         if(leftTime < oldestTime)
           {
            ObjectDelete(0, objName);
            string labelName = Prefix + "Label_" + StringSubstr(objName, StringLen(Prefix));
            ObjectDelete(0, labelName);
           }
        }
     }
  }

//+------------------------------------------------------------------+
//| Send alert notification                                          |
//+------------------------------------------------------------------+
void SendAlert(int direction, datetime barTime, double high, double low)
  {
   string dirStr = (direction == 1) ? "BULLISH" : ((direction == -1) ? "BEARISH" : "NEUTRAL");
   string msg = StringFormat("Inside Bar %s | %s | Range: %.5f - %.5f", dirStr, TimeToString(barTime), low, high);
   Alert(msg);
   SendNotification(msg);
   PlaySound("alert.wav");
  }

//+------------------------------------------------------------------+
//| Determine direction based on small candle close vs open          |
//+------------------------------------------------------------------+
int GetDirection(double open, double close)
  {
   if(close > open)
      return 1;   // bullish
   if(close < open)
      return -1;  // bearish
   return 0;                    // neutral
  }

//+------------------------------------------------------------------+
//| Get color based on direction (for auto mode)                     |
//+------------------------------------------------------------------+
color GetColorByDirection(int direction)
  {
   if(direction == 1)
      return BullishColor;
   if(direction == -1)
      return BearishColor;
   return NeutralColor;
  }

//+------------------------------------------------------------------+
//| Custom indicator iteration function                               |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {
   if(time[rates_total-1] == lastBarTime)
      return(rates_total);
   lastBarTime = time[rates_total-1];

// Resize alert tracking array
   if(ArraySize(alertSentForBar) != rates_total)
     {
      ArrayResize(alertSentForBar, rates_total);
      ArrayInitialize(alertSentForBar, false);
     }

   DeleteOldRectangles(time[rates_total-1]);

   int startBar = MathMax(0, rates_total - actualBarsToKeep);
   int lastPatternBar = -1;

// Find latest pattern for label (if LabelOnlyOnNewBar is enabled)
   if(LabelOnlyOnNewBar && EnableLabel)
     {
      for(int i = rates_total - 1; i >= startBar + 2; i--)
        {
         int smallIdx = i - 1;
         int bigIdx   = i - 2;
         if(high[smallIdx] <= high[bigIdx] && low[smallIdx] >= low[bigIdx])
           {
            int dir = GetDirection(open[smallIdx], close[smallIdx]);
            if(!FilterDirection || dir != 0)
              {
               lastPatternBar = bigIdx;
               break;
              }
           }
        }
     }

// Main loop - detect inside bars
   for(int i = rates_total - 1; i >= startBar + 2; i--)
     {
      int smallIdx = i - 1;
      int bigIdx   = i - 2;

      // Inside bar condition
      if(high[smallIdx] <= high[bigIdx] && low[smallIdx] >= low[bigIdx])
        {
         int direction = GetDirection(open[smallIdx], close[smallIdx]);
         if(FilterDirection && direction == 0)
            continue;

         int targetIdx = bigIdx + actualForwardBars;
         if(targetIdx >= rates_total)
            targetIdx = rates_total - 1;

         string objName = Prefix + IntegerToString(time[bigIdx]);

         // Determine colors
         color borderColor, bgColor;
         if(AutoColorByDirection)
           {
            borderColor = GetColorByDirection(direction);
            bgColor = borderColor;   // same color for fill
           }
         else
           {
            borderColor = NeutralColor;
            bgColor = RectangleFillColor;
           }

         // Create rectangle if not exists
         if(ObjectFind(0, objName) == -1)
           {
            if(ObjectCreate(0, objName, OBJ_RECTANGLE, 0, 0, 0, 0, 0))
              {
               datetime leftTime  = time[bigIdx];
               datetime rightTime = time[targetIdx] + PeriodSeconds();
               double   highLevel = high[bigIdx];
               double   lowLevel  = low[bigIdx];

               ObjectSetInteger(0, objName, OBJPROP_TIME, 0, leftTime);
               ObjectSetInteger(0, objName, OBJPROP_TIME, 1, rightTime);
               ObjectSetDouble(0, objName, OBJPROP_PRICE, 0, highLevel);
               ObjectSetDouble(0, objName, OBJPROP_PRICE, 1, lowLevel);

               // Border color
               ObjectSetInteger(0, objName, OBJPROP_COLOR, (long)borderColor);

               // Fill color (background)
               if(RectangleFill)
                  ObjectSetInteger(0, objName, OBJPROP_BGCOLOR, (long)bgColor);
               else
                  ObjectSetInteger(0, objName, OBJPROP_BGCOLOR, (long)clrNONE);

               // Ensure background is drawn behind chart data
               ObjectSetInteger(0, objName, OBJPROP_BACK, true);
               ObjectSetInteger(0, objName, OBJPROP_WIDTH, 1);
               ObjectSetInteger(0, objName, OBJPROP_STYLE, STYLE_SOLID);
               ObjectSetInteger(0, objName, OBJPROP_SELECTABLE, false);
               ObjectSetInteger(0, objName, OBJPROP_HIDDEN, false);
              }
           }

         // Text label
         if(EnableLabel)
           {
            bool drawLabel = (!LabelOnlyOnNewBar || bigIdx == lastPatternBar);
            if(drawLabel)
              {
               string labelName = Prefix + "Label_" + IntegerToString(time[bigIdx]);
               if(ObjectFind(0, labelName) == -1)
                 {
                  datetime labelTime = time[targetIdx] + PeriodSeconds() * 2;
                  double labelPrice = high[bigIdx] + (high[bigIdx] - low[bigIdx]) * 0.2;
                  if(ObjectCreate(0, labelName, OBJ_TEXT, 0, labelTime, labelPrice))
                    {
                     string dirText = "";
                     if(FilterDirection || AutoColorByDirection)
                        dirText = (direction == 1) ? " ⬆ BULLISH" : ((direction == -1) ? " ⬇ BEARISH" : "");
                     string text = "Inside Bar" + dirText;
                     ObjectSetString(0, labelName, OBJPROP_TEXT, text);
                     ObjectSetInteger(0, labelName, OBJPROP_FONTSIZE, 9);
                     ObjectSetInteger(0, labelName, OBJPROP_COLOR, (long)borderColor);
                     ObjectSetInteger(0, labelName, OBJPROP_BACK, false);
                     ObjectSetInteger(0, labelName, OBJPROP_SELECTABLE, false);
                    }
                 }
              }
           }

         // Send alert only once per pattern
         if(EnableAlert && !alertSentForBar[bigIdx])
           {
            alertSentForBar[bigIdx] = true;
            SendAlert(direction, time[bigIdx], high[bigIdx], low[bigIdx]);
           }
        }
     }

   return(rates_total);
  }
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+

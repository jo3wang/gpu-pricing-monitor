# Setting up Automated Daily Data Collection

This guide explains how to set up automated daily data collection for the GPU pricing monitor.

## Method 1: Using Cron (Recommended for macOS/Linux)

1. **Edit your crontab:**
   ```bash
   crontab -e
   ```

2. **Add the following line to run daily at 6:00 AM:**
   ```cron
   0 6 * * * cd /Users/joewang/Desktop/recession\ tracker/gpu-pricing-monitor && ./daily_update.sh >> daily_update.log 2>&1
   ```

3. **Alternative times:**
   - Every 6 hours: `0 */6 * * * cd /Users/joewang/Desktop/recession\ tracker/gpu-pricing-monitor && ./daily_update.sh >> daily_update.log 2>&1`
   - Every 12 hours: `0 */12 * * * cd /Users/joewang/Desktop/recession\ tracker/gpu-pricing-monitor && ./daily_update.sh >> daily_update.log 2>&1`
   - Daily at 9 AM: `0 9 * * * cd /Users/joewang/Desktop/recession\ tracker/gpu-pricing-monitor && ./daily_update.sh >> daily_update.log 2>&1`

## Method 2: Manual Execution

Simply run the script manually whenever you want to update the data:

```bash
cd "/Users/joewang/Desktop/recession tracker/gpu-pricing-monitor"
./daily_update.sh
```

## Method 3: Using launchd (macOS Alternative)

Create a launch agent for more reliable scheduling on macOS:

1. **Create the plist file:**
   ```bash
   mkdir -p ~/Library/LaunchAgents
   ```

2. **Create ~/Library/LaunchAgents/com.user.gpu-pricing-monitor.plist:**
   ```xml
   <?xml version="1.0" encoding="UTF-8"?>
   <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
   <plist version="1.0">
   <dict>
       <key>Label</key>
       <string>com.user.gpu-pricing-monitor</string>
       <key>ProgramArguments</key>
       <array>
           <string>/Users/joewang/Desktop/recession tracker/gpu-pricing-monitor/daily_update.sh</string>
       </array>
       <key>WorkingDirectory</key>
       <string>/Users/joewang/Desktop/recession tracker/gpu-pricing-monitor</string>
       <key>StartCalendarInterval</key>
       <dict>
           <key>Hour</key>
           <integer>6</integer>
           <key>Minute</key>
           <integer>0</integer>
       </dict>
       <key>StandardOutPath</key>
       <string>/Users/joewang/Desktop/recession tracker/gpu-pricing-monitor/daily_update.log</string>
       <key>StandardErrorPath</key>
       <string>/Users/joewang/Desktop/recession tracker/gpu-pricing-monitor/daily_update.log</string>
   </dict>
   </plist>
   ```

3. **Load the launch agent:**
   ```bash
   launchctl load ~/Library/LaunchAgents/com.user.gpu-pricing-monitor.plist
   ```

## Monitoring and Troubleshooting

- **Check logs:** `tail -f daily_update.log`
- **Test the script:** `./daily_update.sh`
- **Check if cron job is running:** `crontab -l`
- **View cron logs on macOS:** `log show --predicate 'eventMessage contains "cron"' --last 1d`

## What the Automation Does

1. **Collects fresh data** from AWS and Azure APIs
2. **Processes and cleans** the raw pricing data
3. **Computes stress index** for pricing volatility analysis
4. **Updates** the dashboard data file (`gpu_pricing_summary.csv`)
5. **Logs** all operations for monitoring

After setup, your dashboard will always show the latest pricing data!
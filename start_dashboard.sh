#!/bin/bash

# GPU Pricing Monitor Dashboard Startup Script
# This script provides multiple ways to start the Streamlit dashboard

echo "=== GPU Pricing Monitor Dashboard Startup ==="
echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"
echo "Streamlit version: $(streamlit --version)"
echo ""

# Function to check if port is already in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        echo "Port $1 is already in use. Checking if it's our Streamlit app..."
        if curl -s -f http://localhost:$1 >/dev/null; then
            echo "Dashboard is already running at http://localhost:$1"
            echo "Opening in browser..."
            open http://localhost:$1
            return 0
        else
            echo "Port $1 is occupied by another service. Please stop it or use a different port."
            return 1
        fi
    fi
    return 2
}

# Function to start dashboard with specific settings
start_dashboard() {
    local port=${1:-8501}
    local host=${2:-localhost}
    
    echo "Starting dashboard on $host:$port..."
    echo "Access URL: http://$host:$port"
    echo ""
    echo "Press Ctrl+C to stop the dashboard"
    echo "----------------------------------------"
    
    # Start streamlit with explicit settings
    streamlit run dashboard.py \
        --server.address="$host" \
        --server.port="$port" \
        --server.headless=false \
        --browser.gatherUsageStats=false \
        --logger.level=info
}

# Main execution
case "${1:-default}" in
    "check")
        echo "Checking if dashboard is already running..."
        check_port 8501
        exit $?
        ;;
    "kill")
        echo "Stopping any existing dashboard processes..."
        pkill -f "streamlit run dashboard.py"
        sleep 2
        echo "Processes stopped."
        ;;
    "localhost")
        start_dashboard 8501 "localhost"
        ;;
    "all")
        start_dashboard 8501 "0.0.0.0"
        ;;
    "port")
        if [ -z "$2" ]; then
            echo "Usage: $0 port <port_number>"
            exit 1
        fi
        start_dashboard "$2" "localhost"
        ;;
    "help")
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  default/localhost  - Start on localhost:8501 (default)"
        echo "  all               - Start on all interfaces (0.0.0.0:8501)"
        echo "  port <number>     - Start on specified port"
        echo "  check             - Check if dashboard is already running"
        echo "  kill              - Stop any running dashboard processes"
        echo "  help              - Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0                # Start on localhost:8501"
        echo "  $0 all            # Start on all interfaces"
        echo "  $0 port 8080      # Start on localhost:8080"
        echo "  $0 check          # Check if running"
        echo "  $0 kill           # Stop dashboard"
        ;;
    *)
        # Default behavior - start on localhost
        if check_port 8501; then
            exit 0
        fi
        start_dashboard 8501 "localhost"
        ;;
esac
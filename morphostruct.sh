#!/bin/bash

# MorphoStruct - Unified Run Script
# Manages backend (FastAPI) and frontend (Next.js) development servers

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Project paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
VENV_DIR="$BACKEND_DIR/venv"

# PIDs for cleanup
BACKEND_PID=""
FRONTEND_PID=""
PID_FILE="$SCRIPT_DIR/.morphostruct-pids"

# Print colored message
print_msg() {
    local color=$1
    local msg=$2
    echo -e "${color}${msg}${NC}"
}

print_success() { print_msg "$GREEN" "✓ $1"; }
print_error() { print_msg "$RED" "✗ $1"; }
print_info() { print_msg "$BLUE" "ℹ $1"; }
print_warning() { print_msg "$YELLOW" "⚠ $1"; }
print_header() { print_msg "$CYAN" "==== $1 ===="; }

# Show usage
show_help() {
    cat << EOF
MorphoStruct - Unified Run Script

Usage: $0 [COMMAND]

Commands:
    (no args)       Start both backend and frontend servers
    backend         Start only the backend server (FastAPI on :8000)
    frontend        Start only the frontend server (Next.js on :3000)
    stop            Stop all running MorphoStruct servers
    install         Install all dependencies (backend + frontend)
    --help, -h      Show this help message

Examples:
    $0              # Start both servers
    $0 backend      # Start only backend
    $0 install      # Install all dependencies

The script will:
  • Create Python virtual environment if needed
  • Install missing dependencies automatically
  • Check for port conflicts
  • Handle graceful shutdown on Ctrl+C
  • Show colored status output

EOF
    exit 0
}

# Cleanup on exit
cleanup() {
    print_header "Shutting down services"

    if [ -n "$BACKEND_PID" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        print_info "Stopping backend (PID: $BACKEND_PID)"
        kill -TERM "$BACKEND_PID" 2>/dev/null || true
    fi

    if [ -n "$FRONTEND_PID" ] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        print_info "Stopping frontend (PID: $FRONTEND_PID)"
        kill -TERM "$FRONTEND_PID" 2>/dev/null || true
    fi

    # Wait a moment for graceful shutdown
    sleep 1

    # Force kill if still running
    if [ -n "$BACKEND_PID" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        kill -9 "$BACKEND_PID" 2>/dev/null || true
    fi

    if [ -n "$FRONTEND_PID" ] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        kill -9 "$FRONTEND_PID" 2>/dev/null || true
    fi

    print_success "All services stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# Save PIDs to file for later stop command
save_pids() {
    echo "BACKEND_PID=$BACKEND_PID" > "$PID_FILE"
    echo "FRONTEND_PID=$FRONTEND_PID" >> "$PID_FILE"
}

# Stop servers using saved PIDs (for stop command)
stop_servers() {
    print_header "Stopping MorphoStruct Servers"

    # Try to read saved PIDs
    if [ -f "$PID_FILE" ]; then
        source "$PID_FILE"
    fi

    # Kill by PID if available
    local killed=false

    if [ -n "$BACKEND_PID" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        print_info "Stopping backend (PID: $BACKEND_PID)"
        kill -TERM "$BACKEND_PID" 2>/dev/null || true
        killed=true
    fi

    if [ -n "$FRONTEND_PID" ] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        print_info "Stopping frontend (PID: $FRONTEND_PID)"
        kill -TERM "$FRONTEND_PID" 2>/dev/null || true
        killed=true
    fi

    # Also kill by port as fallback
    if lsof -ti:8000 >/dev/null 2>&1; then
        print_info "Stopping process on port 8000"
        lsof -ti:8000 | xargs kill -9 2>/dev/null || true
        killed=true
    fi

    if lsof -ti:3000 >/dev/null 2>&1; then
        print_info "Stopping process on port 3000"
        lsof -ti:3000 | xargs kill -9 2>/dev/null || true
        killed=true
    fi

    # Clean up PID file
    rm -f "$PID_FILE"

    if [ "$killed" = true ]; then
        print_success "All MorphoStruct services stopped"
    else
        print_info "No running MorphoStruct services found"
    fi
}

# Check if port is in use
check_port() {
    local port=$1
    local service=$2

    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        print_error "Port $port is already in use (required for $service)"
        print_info "Please stop the process using port $port or run:"
        print_info "  lsof -ti:$port | xargs kill -9"
        exit 1
    fi
}

# Check and create Python virtual environment
setup_backend_venv() {
    print_header "Checking Backend Virtual Environment"

    if [ ! -d "$VENV_DIR" ]; then
        print_info "Creating Python virtual environment..."
        python3 -m venv "$VENV_DIR"
        print_success "Virtual environment created"
    else
        print_success "Virtual environment exists"
    fi
}

# Install backend dependencies
install_backend() {
    print_header "Installing Backend Dependencies"

    setup_backend_venv

    if [ ! -f "$BACKEND_DIR/requirements.txt" ]; then
        print_error "requirements.txt not found in backend directory"
        exit 1
    fi

    print_info "Installing Python packages..."
    cd "$BACKEND_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip > /dev/null 2>&1
    pip install -r requirements.txt
    print_success "Backend dependencies installed"
}

# Check if backend dependencies are installed
check_backend_deps() {
    if [ ! -d "$VENV_DIR" ]; then
        return 1
    fi

    source "$VENV_DIR/bin/activate"
    python -c "import fastapi" 2>/dev/null || return 1
    return 0
}

# Install frontend dependencies
install_frontend() {
    print_header "Installing Frontend Dependencies"

    if [ ! -f "$FRONTEND_DIR/package.json" ]; then
        print_error "package.json not found in frontend directory"
        exit 1
    fi

    print_info "Installing Node.js packages..."
    cd "$FRONTEND_DIR"
    npm install
    print_success "Frontend dependencies installed"
}

# Check if frontend dependencies are installed
check_frontend_deps() {
    [ -d "$FRONTEND_DIR/node_modules" ] && return 0 || return 1
}

# Install all dependencies
install_all() {
    install_backend
    install_frontend
    print_success "All dependencies installed successfully"
}

# Start backend server
start_backend() {
    print_header "Starting Backend Server"

    # Check dependencies
    if ! check_backend_deps; then
        print_warning "Backend dependencies not found"
        install_backend
    fi

    # Check port
    check_port 8000 "backend"

    # Check if run.py exists
    if [ ! -f "$BACKEND_DIR/run.py" ]; then
        print_error "run.py not found in backend directory"
        exit 1
    fi

    print_info "Starting FastAPI on http://localhost:8000"
    cd "$BACKEND_DIR"
    source "$VENV_DIR/bin/activate"
    python run.py > /tmp/morphostruct-backend.log 2>&1 &
    BACKEND_PID=$!

    # Wait a moment and check if it started successfully
    sleep 2
    if kill -0 "$BACKEND_PID" 2>/dev/null; then
        print_success "Backend server started (PID: $BACKEND_PID)"
        print_info "Backend logs: tail -f /tmp/morphostruct-backend.log"
    else
        print_error "Backend server failed to start"
        print_info "Check logs: cat /tmp/morphostruct-backend.log"
        exit 1
    fi
}

# Start frontend server
start_frontend() {
    print_header "Starting Frontend Server"

    # Check dependencies
    if ! check_frontend_deps; then
        print_warning "Frontend dependencies not found"
        install_frontend
    fi

    # Check port
    check_port 3000 "frontend"

    print_info "Starting Next.js on http://localhost:3000"
    cd "$FRONTEND_DIR"
    npm run dev > /tmp/morphostruct-frontend.log 2>&1 &
    FRONTEND_PID=$!

    # Wait a moment and check if it started successfully
    sleep 3
    if kill -0 "$FRONTEND_PID" 2>/dev/null; then
        print_success "Frontend server started (PID: $FRONTEND_PID)"
        print_info "Frontend logs: tail -f /tmp/morphostruct-frontend.log"
    else
        print_error "Frontend server failed to start"
        print_info "Check logs: cat /tmp/morphostruct-frontend.log"
        exit 1
    fi
}

# Open browser (cross-platform)
open_browser() {
    local url=$1
    sleep 2  # Wait for server to be ready

    if command -v open &> /dev/null; then
        # macOS
        open "$url"
    elif command -v xdg-open &> /dev/null; then
        # Linux
        xdg-open "$url"
    elif command -v start &> /dev/null; then
        # Windows (Git Bash)
        start "$url"
    else
        print_warning "Could not auto-open browser. Please visit: $url"
    fi
}

# Start both servers
start_both() {
    print_header "Starting MorphoStruct Development Environment"

    start_backend
    start_frontend

    echo ""
    print_header "Services Running"
    print_success "Backend:  http://localhost:8000"
    print_success "Frontend: http://localhost:3000"
    print_success "API Docs: http://localhost:8000/docs"
    echo ""

    # Save PIDs for stop command
    save_pids

    # Auto-open browser
    print_info "Opening browser..."
    open_browser "http://localhost:3000"

    print_info "Press Ctrl+C to stop all services"
    print_info "Or run: ./shed.sh stop"

    # Wait for user interrupt
    wait
}

# Main script logic
main() {
    # Check if running from correct directory
    if [ ! -d "$BACKEND_DIR" ] || [ ! -d "$FRONTEND_DIR" ]; then
        print_error "This script must be run from the MorphoStruct project root directory"
        print_info "Expected structure:"
        print_info "  ./backend/  (FastAPI project)"
        print_info "  ./frontend/ (Next.js project)"
        exit 1
    fi

    case "${1:-}" in
        "")
            start_both
            ;;
        "backend")
            start_backend
            print_info "Press Ctrl+C to stop backend"
            wait
            ;;
        "frontend")
            start_frontend
            print_info "Press Ctrl+C to stop frontend"
            wait
            ;;
        "install")
            install_all
            ;;
        "stop")
            stop_servers
            exit 0
            ;;
        "--help"|"-h"|"help")
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            echo ""
            show_help
            ;;
    esac
}

main "$@"

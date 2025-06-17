import subprocess
import signal
import psutil
import time
import os
import streamlit as st

def is_main_py_running():
    """Check if main.py is currently running - optimized version"""
    try:
        # Method 1: Use pgrep on Unix systems (fastest)
        if os.name != 'nt':  # Unix/Linux/macOS
            try:
                result = subprocess.run(['pgrep', '-f', 'main.py'], 
                                      capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    pids = result.stdout.strip().split('\n')
                    return True, int(pids[0]) if pids[0] else None
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        
        # Method 2: Use tasklist on Windows (faster than psutil iteration)
        elif os.name == 'nt':  # Windows
            try:
                result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'], 
                                      capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')[1:]  # Skip header
                    for line in lines:
                        if 'python.exe' in line:
                            # Extract PID from CSV format
                            parts = line.split('","')
                            if len(parts) >= 2:
                                pid = int(parts[1].replace('"', ''))
                                # Quick check if this PID is running main.py
                                try:
                                    proc = psutil.Process(pid)
                                    cmdline = ' '.join(proc.cmdline())
                                    if 'main.py' in cmdline:
                                        return True, pid
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    continue
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        
        # Method 3: Cached PID check (fastest for repeated calls)
        if hasattr(st.session_state, 'last_known_pid') and st.session_state.last_known_pid:
            try:
                proc = psutil.Process(st.session_state.last_known_pid)
                if proc.is_running():
                    cmdline = ' '.join(proc.cmdline())
                    if 'main.py' in cmdline:
                        return True, st.session_state.last_known_pid
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                st.session_state.last_known_pid = None
        
        # Method 4: Optimized psutil fallback (only if other methods fail)
        # Only check python processes, not all processes
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if proc.info['name'] and 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.cmdline())
                    if 'main.py' in cmdline:
                        return True, proc.info['pid']
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
                
    except Exception:
        pass
    
    return False, None

def start_main_py():
    """Start main.py in a new terminal window"""
    try:
        import platform
        import sys
        system = platform.system()
        
        if system == "Windows":
            # Windows - use specific Python executable path and & operator
            python_exe = sys.executable  # Gets the current Python executable path
            main_py_path = os.path.join(os.getcwd(), "main.py")
            
            # Use & operator to run in background and open new cmd window
            cmd = f'start cmd /k "{python_exe} {main_py_path}"'
            process = subprocess.Popen(cmd, shell=True, cwd=os.getcwd())
            
        elif system == "Darwin":  # macOS
            # macOS - open new Terminal window
            python_exe = sys.executable
            main_py_path = os.path.join(os.getcwd(), "main.py")
            script = f'tell application "Terminal" to do script "cd {os.getcwd()} && {python_exe} {main_py_path}"'
            process = subprocess.Popen(['osascript', '-e', script])
            
        else:  # Linux
            # Try different terminal emulators in order of preference
            python_exe = sys.executable
            main_py_path = os.path.join(os.getcwd(), "main.py")
            
            terminals = [
                ['gnome-terminal', '--', 'bash', '-c', f'cd {os.getcwd()} && {python_exe} {main_py_path}; exec bash'],
                ['xterm', '-e', f'cd {os.getcwd()} && {python_exe} {main_py_path}; exec bash'],
                ['konsole', '-e', f'cd {os.getcwd()} && {python_exe} {main_py_path}; exec bash'],
                ['x-terminal-emulator', '-e', f'cd {os.getcwd()} && {python_exe} {main_py_path}; exec bash']
            ]
            
            process = None
            for terminal_cmd in terminals:
                try:
                    process = subprocess.Popen(terminal_cmd, cwd=os.getcwd())
                    break
                except FileNotFoundError:
                    continue
            
            if process is None:
                # Fallback to background process if no terminal found
                process = subprocess.Popen([python_exe, main_py_path], 
                                         cwd=os.getcwd(),
                                         stdout=subprocess.PIPE, 
                                         stderr=subprocess.PIPE)
        
        # Give the process a moment to start
        
        time.sleep(1)
        
        # Try to find the actual main.py process (not the terminal process)
        running, actual_pid = is_main_py_running()
        if running:
            return True, actual_pid
        else:
            return True, process.pid
            
    except Exception as e:
        return False, str(e)

def stop_main_py():
    """Stop main.py process"""
    running, pid = is_main_py_running()
    if running:
        try:
            # Try to terminate gracefully first
            process = psutil.Process(pid)
            process.terminate()
            
            # Wait a bit for graceful shutdown
            try:
                process.wait(timeout=5)
            except psutil.TimeoutExpired:
                # Force kill if graceful shutdown didn't work
                process.kill()
            
            return True, "Process stopped successfully"
        except Exception as e:
            return False, str(e)
    else:
        return False, "Process not running"
    

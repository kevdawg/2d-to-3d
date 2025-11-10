#!/usr/bin/env python3
"""
Timing Tracker Module
Provides clean, consistent timing display for pipeline steps.
"""
import time
from contextlib import contextmanager
from typing import Optional, Dict, List
import sys


class TimingTracker:
    """
    Tracks execution time for pipeline steps with hierarchical display.
    
    Usage:
        tracker = TimingTracker(total_steps=4)
        
        with tracker.step(1, "Background Removal"):
            # do work
            tracker.substep("API call")
            # more work
            tracker.substep("Auto-crop")
        
        tracker.print_summary()
    """
    
    def __init__(self, total_steps: int = 1, name: str = "Pipeline", 
                 ok_symbol: str = "[OK]", warn_symbol: str = "[!]"):
        self.total_steps = total_steps
        self.name = name
        self.ok_symbol = ok_symbol
        self.warn_symbol = warn_symbol
        self.steps: List[Dict] = []
        self.current_step: Optional[Dict] = None
        self.start_time = time.time()
        
    @contextmanager
    def step(self, step_num: int, step_name: str, show_progress: bool = True):
        """
        Context manager for timing a major pipeline step.
        
        Args:
            step_num: Step number (1-based)
            step_name: Display name of the step
            show_progress: Show progress indicator
        """
        # Start step
        step_data = {
            'number': step_num,
            'name': step_name,
            'start_time': time.time(),
            'end_time': None,
            'duration': None,
            'substeps': []
        }
        self.current_step = step_data
        
        # Display header
        if show_progress:
            bar_width = 40
            progress_pct = int(step_num / self.total_steps * 100)
            filled = int(bar_width * step_num / self.total_steps)
            bar = '█' * filled + '░' * (bar_width - filled)
            
            print(f"\n┌─────────────────────────────────────────────────────────┐")
            print(f"│ [{step_num}/{self.total_steps}] {step_name:<44} {progress_pct:>3}% │")
            print(f"│ {bar} │")
            print(f"└─────────────────────────────────────────────────────────┘")
        else:
            print(f"\n[{step_num}/{self.total_steps}] {step_name}...")
        
        try:
            yield self
        finally:
            # End step
            step_data['end_time'] = time.time()
            step_data['duration'] = step_data['end_time'] - step_data['start_time']
            self.steps.append(step_data)
            
            # Display completion
            duration_str = self._format_duration(step_data['duration'])
            print(f"  └─ Completed in {duration_str} {self.ok_symbol}")
            
            self.current_step = None
    
    def substep(self, substep_name: str, details: str = ""):
        """
        Log a substep within the current step (no timing, just progress indicator).
        
        Args:
            substep_name: Name of the substep
            details: Optional details to display
        """
        if self.current_step is not None:
            timestamp = time.time() - self.current_step['start_time']
            self.current_step['substeps'].append({
                'name': substep_name,
                'details': details,
                'timestamp': timestamp
            })
            
            if details:
                print(f"  ├─ {substep_name}: {details}")
            else:
                print(f"  ├─ {substep_name}")
    
    def substep_timed(self, substep_name: str) -> 'SubstepTimer':
        """
        Create a timed substep (for detailed profiling).
        
        Returns:
            SubstepTimer context manager
        """
        return SubstepTimer(self, substep_name)
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration as human-readable string."""
        if seconds < 1:
            return f"{int(seconds * 1000)}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        else:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs:02d}s"
    
    def print_summary(self, output_info: str = ""):
        """
        Print a comprehensive timing summary with bottleneck identification.
        
        Args:
            output_info: Additional info to display (e.g., output folder)
        """
        total_duration = time.time() - self.start_time
        
        print("\n" + "="*60)
        print(f"{self.ok_symbol} COMPLETE - {self.name}")
        print("="*60)
        
        if output_info:
            print(f"  📁 {output_info}")
        
        print(f"\n  Total time: {self._format_duration(total_duration)}")
        
        # Step breakdown
        print("\n  Step Breakdown:")
        
        # Find bottleneck (longest step)
        bottleneck_idx = None
        max_duration = 0
        for i, step in enumerate(self.steps):
            if step['duration'] > max_duration:
                max_duration = step['duration']
                bottleneck_idx = i
        
        for i, step in enumerate(self.steps):
            duration_str = self._format_duration(step['duration'])
            percentage = (step['duration'] / total_duration * 100) if total_duration > 0 else 0
            
            # Format step line
            is_last = (i == len(self.steps) - 1)
            prefix = "└─" if is_last else "├─"
            
            # Add bottleneck indicator
            bottleneck_mark = f" {self.warn_symbol} BOTTLENECK" if i == bottleneck_idx and percentage > 50 else ""
            
            print(f"  {prefix} [{step['number']}/{self.total_steps}] {step['name']:<25} {duration_str:>8} ({percentage:>5.1f}%){bottleneck_mark}")
            
            # Show detailed substeps if available and step is bottleneck
            if i == bottleneck_idx and step['substeps'] and percentage > 50:
                for substep in step['substeps']:
                    if 'duration' in substep:
                        substep_dur = self._format_duration(substep['duration'])
                        print(f"      ├─ {substep['name']:<20} {substep_dur:>8}")
        
        print("="*60 + "\n")
    
    def print_batch_summary(self, completed: int, total: int, avg_time: float):
        """
        Print batch processing summary.
        
        Args:
            completed: Number of images completed
            total: Total images to process
            avg_time: Average time per image
        """
        remaining = total - completed
        est_remaining = remaining * avg_time
        
        progress_pct = int(completed / total * 100)
        bar_width = 50
        filled = int(bar_width * completed / total)
        bar = '█' * filled + '░' * (bar_width - filled)
        
        print("\n" + "="*60)
        print(f"  BATCH PROGRESS: {completed}/{total} images ({progress_pct}%)")
        print("="*60)
        print(f"  {bar}")
        print(f"\n  Statistics:")
        print(f"  ├─ Completed: {completed} images (avg {self._format_duration(avg_time)} each)")
        print(f"  ├─ Remaining: {remaining} images")
        print(f"  └─ Estimated time remaining: {self._format_duration(est_remaining)}")
        print("="*60 + "\n")


class SubstepTimer:
    """Context manager for timing substeps within a step."""
    
    def __init__(self, tracker: TimingTracker, substep_name: str):
        self.tracker = tracker
        self.substep_name = substep_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if self.tracker.current_step is not None:
            self.tracker.current_step['substeps'].append({
                'name': self.substep_name,
                'duration': duration,
                'timestamp': self.start_time - self.tracker.current_step['start_time']
            })
            
            duration_str = self.tracker._format_duration(duration)
            print(f"  ├─ {self.substep_name} ... {duration_str}")


# Convenience function for simple timing without full tracker
@contextmanager
def simple_timer(label: str):
    """
    Simple timer for one-off timing needs.
    
    Usage:
        with simple_timer("Loading model"):
            model = load_model()
    """
    print(f"{label}...", end="", flush=True)
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        if duration < 1:
            print(f" {int(duration * 1000)}ms {self.ok_symbol}")
        elif duration < 60:
            print(f" {duration:.1f}s {self.ok_symbol}")
        else:
            mins = int(duration // 60)
            secs = int(duration % 60)
            print(f" {mins}m {secs:02d}s {self.ok_symbol}")
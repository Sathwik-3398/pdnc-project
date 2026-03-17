"""
attendance_stats.py
"""
import os
import pandas as pd

def get_student_list():
    dataset_dir = 'dataset'
    if not os.path.exists(dataset_dir):
        return []
    return [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith('.')]

def get_attendance_stats():
    attendance_dir = 'attendance'
    if not os.path.exists(attendance_dir):
        return []

    all_files = [f for f in os.listdir(attendance_dir) if f.endswith('.csv')]
    if not all_files:
        return []

    total_dates = len(all_files)
    student_counts = {s: 0 for s in get_student_list()}

    for file in all_files:
        try:
            df = pd.read_csv(os.path.join(attendance_dir, file))
            for name in df['Name'].unique():
                if name in student_counts:
                    student_counts[name] += 1
        except:
            continue

    stats = []
    for name, count in student_counts.items():
        percentage = (count / total_dates * 100) if total_dates > 0 else 0
        stats.append({
            "name": name,
            "present": count,
            "total": total_dates,
            "percentage": f"{percentage:.1f}%"
        })
    return stats

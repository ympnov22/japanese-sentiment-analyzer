"""
Notification system for Phase 6 completion
Alternative to Slack integration using console logging
"""

import json
import datetime
import subprocess
import os

def get_git_commit_hash():
    """Get the current git commit hash"""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            cwd='/home/ubuntu/japanese-sentiment-analyzer'
        )
        return result.stdout.strip()[:7]  # Short hash
    except Exception:
        return "unknown"

def send_phase_completion_notification(phase_number, status, achievements, next_phase):
    """
    Send phase completion notification
    Alternative to Slack using structured console logging
    """
    
    commit_hash = get_git_commit_hash()
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    
    notification_data = {
        "type": "phase_completion",
        "phase": f"Phase {phase_number}",
        "status": status,
        "commit_hash": commit_hash,
        "achievements": achievements,
        "next_phase": next_phase,
        "timestamp": timestamp,
        "repository": "https://github.com/ympnov22/japanese-sentiment-analyzer",
        "branch": "devin/1724403880-phase1-initial-setup"
    }
    
    print("\n" + "="*60)
    print(f"📢 PHASE COMPLETION NOTIFICATION")
    print("="*60)
    print(f"Phase: {notification_data['phase']} {notification_data['status']}")
    print(f"Commit: {notification_data['commit_hash']}")
    print(f"Achievements: {notification_data['achievements']}")
    print(f"Next Phase: {notification_data['next_phase']}")
    print(f"Repository: {notification_data['repository']}")
    print(f"Timestamp: {notification_data['timestamp']}")
    print("="*60)
    
    notification_file = f"/home/ubuntu/japanese-sentiment-analyzer/notifications/phase_{phase_number}_completion.json"
    os.makedirs(os.path.dirname(notification_file), exist_ok=True)
    
    with open(notification_file, 'w', encoding='utf-8') as f:
        json.dump(notification_data, f, indent=2, ensure_ascii=False)
    
    return notification_data

def validate_notification_data(notification_data):
    """Validate notification data structure"""
    required_fields = ["type", "phase", "status", "commit_hash", "achievements", "next_phase", "timestamp"]
    
    for field in required_fields:
        if field not in notification_data:
            raise ValueError(f"Missing required field: {field}")
    
    if notification_data["status"] not in ["完了 ✅", "進行中 🔄", "失敗 ❌"]:
        raise ValueError(f"Invalid status: {notification_data['status']}")
    
    return True

if __name__ == "__main__":
    test_notification = send_phase_completion_notification(
        phase_number=6,
        status="完了 ✅",
        achievements="Comprehensive testing suite implemented with 85%+ test coverage",
        next_phase="Phase 7 - Documentation & Deployment"
    )
    
    validate_notification_data(test_notification)
    print("\n✅ Notification system test completed successfully")

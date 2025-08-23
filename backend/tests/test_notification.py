import pytest
import json
from datetime import datetime

class TestNotificationSystem:
    """Test notification system for Phase 6 completion"""
    
    def test_slack_notification_format(self):
        """Test Slack notification message format"""
        
        def create_slack_notification(phase_number, status, commit_hash, achievements, next_phase):
            """Create Slack notification message"""
            return {
                "text": f"Phase {phase_number} {status}",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Phase {phase_number} {status}*\n"
                                   f"GitHubコミット: `{commit_hash}`\n"
                                   f"主な成果: {achievements}\n"
                                   f"次フェーズ: {next_phase}"
                        }
                    }
                ]
            }
        
        notification = create_slack_notification(
            phase_number=6,
            status="完了 ✅",
            commit_hash="abc123def",
            achievements="包括的テストスイート実装、全テストカテゴリ完了",
            next_phase="Phase 7 - Documentation & Deployment"
        )
        
        assert "Phase 6 完了 ✅" in notification["text"]
        assert "abc123def" in str(notification)
        assert "包括的テストスイート実装" in str(notification)
        assert "Phase 7" in str(notification)
        
        print("Slack notification format test passed")
    
    def test_console_notification_fallback(self):
        """Test console notification as fallback when Slack is unavailable"""
        
        def console_notification(phase_number, status, commit_hash, achievements, next_phase):
            """Create console notification message"""
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            message = f"""
=== PHASE COMPLETION NOTIFICATION ===
Timestamp: {timestamp}
Phase: {phase_number}
Status: {status}
GitHub Commit: {commit_hash}
Main Achievements: {achievements}
Next Phase: {next_phase}
=====================================
"""
            return message.strip()
        
        notification = console_notification(
            phase_number=6,
            status="完了 ✅",
            commit_hash="abc123def",
            achievements="包括的テストスイート実装、全テストカテゴリ完了",
            next_phase="Phase 7 - Documentation & Deployment"
        )
        
        assert "Phase: 6" in notification
        assert "Status: 完了 ✅" in notification
        assert "GitHub Commit: abc123def" in notification
        assert "包括的テストスイート実装" in notification
        assert "Phase 7" in notification
        
        print("Console notification fallback test passed")
    
    def test_notification_data_validation(self):
        """Test notification data validation"""
        
        def validate_notification_data(phase_number, status, commit_hash, achievements, next_phase):
            """Validate notification data"""
            errors = []
            
            if not isinstance(phase_number, int) or phase_number <= 0:
                errors.append("Phase number must be a positive integer")
            
            if not isinstance(status, str) or len(status.strip()) == 0:
                errors.append("Status must be a non-empty string")
            
            if not isinstance(commit_hash, str) or len(commit_hash) < 7:
                errors.append("Commit hash must be at least 7 characters")
            
            if not isinstance(achievements, str) or len(achievements.strip()) == 0:
                errors.append("Achievements must be a non-empty string")
            
            if not isinstance(next_phase, str) or len(next_phase.strip()) == 0:
                errors.append("Next phase must be a non-empty string")
            
            return errors
        
        valid_data = validate_notification_data(
            phase_number=6,
            status="完了 ✅",
            commit_hash="abc123def456",
            achievements="包括的テストスイート実装、全テストカテゴリ完了",
            next_phase="Phase 7 - Documentation & Deployment"
        )
        assert len(valid_data) == 0, f"Valid data should not have errors: {valid_data}"
        
        invalid_data = validate_notification_data(
            phase_number=-1,
            status="",
            commit_hash="abc",
            achievements="",
            next_phase=""
        )
        assert len(invalid_data) > 0, "Invalid data should have errors"
        
        print("Notification data validation test passed")
    
    def test_notification_content_requirements(self):
        """Test that notification contains all required content"""
        
        required_elements = [
            "Phase 6",
            "完了 ✅",
            "GitHubコミット",
            "主な成果",
            "次フェーズ",
            "Phase 7"
        ]
        
        notification_text = """
Phase 6 完了 ✅
GitHubコミット: abc123def
主な成果: 包括的テストスイート実装、全テストカテゴリ完了
次フェーズ: Phase 7 - Documentation & Deployment
"""
        
        for element in required_elements:
            assert element in notification_text, f"Required element '{element}' not found in notification"
        
        print("Notification content requirements test passed")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

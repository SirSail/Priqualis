"""Tests for UI structure."""

from streamlit.testing.v1 import AppTest

def test_app_startup():
    """Test that the app starts without error."""
    at = AppTest.from_file("ui/app.py")
    at.run()
    assert not at.exception

def test_pages_load():
    """Test that sidebar navigation exists."""
    at = AppTest.from_file("ui/app.py")
    at.run()
    # Check sidebar title
    assert "Priqualis" in at.sidebar.title[0].value
    # Check navigation radio exists
    assert len(at.sidebar.radio) > 0

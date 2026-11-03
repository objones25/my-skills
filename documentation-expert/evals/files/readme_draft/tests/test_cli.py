from widgetkit import build_widget


def test_build_widget_returns_name():
    assert build_widget("gear") == "gear"

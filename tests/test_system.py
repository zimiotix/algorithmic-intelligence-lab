from ailab.core import system


def test_probe_never_crashes_and_reports_basics():
    info = system.probe()
    assert info.cpu_threads >= 1
    assert info.ram_total_mb > 0
    report = system.format_report(info)
    assert "CPU" in report and "Compute" in report
